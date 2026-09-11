// m1max_metal_limits.swift — runtime Metal capability capture for the M1 Max
// macOS-side pre-install capture 2026-09-11. Prints one JSON object.
// Probes are compile+execute on the real device, not documentation lookups.
// v2: simdgroup_load/store take (matrix&, src, elements_per_row, ulong2);
//     thread ids qualified; atomic expectation counts 32 lanes/threadgroup.
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()
var report = [String: Any]()

guard let device = dev else {
    print("{\"error\": \"no default MTLDevice\"}")
    exit(1)
}

// ---------- 1. Device properties ----------
var props = [String: Any]()
props["name"] = device.name
props["registryID"] = device.registryID
props["headless"] = device.isHeadless
props["lowPower"] = device.isLowPower
props["removable"] = device.isRemovable
props["maxThreadgroupMemoryLength"] = device.maxThreadgroupMemoryLength
props["maxThreadsPerThreadgroup"] = ["width": device.maxThreadsPerThreadgroup.width,
                                     "height": device.maxThreadsPerThreadgroup.height,
                                     "depth": device.maxThreadsPerThreadgroup.depth]
props["hasUnifiedMemory"] = device.hasUnifiedMemory
props["recommendedMaxWorkingSetSize"] = device.recommendedMaxWorkingSetSize
props["maxBufferLength"] = device.maxBufferLength
props["currentAllocatedSize"] = device.currentAllocatedSize
props["argumentBuffersSupport"] = device.argumentBuffersSupport == .tier2 ? "tier2" : (device.argumentBuffersSupport == .tier1 ? "tier1" : "tier0")
props["maxArgumentBufferSamplerCount"] = device.maxArgumentBufferSamplerCount
props["supports32BitFloatFiltering"] = device.supports32BitFloatFiltering
props["supports32BitMSAA"] = device.supports32BitMSAA
props["readWriteTextureTier"] = device.readWriteTextureSupport == .tier2 ? "tier2" : (device.readWriteTextureSupport == .tier1 ? "tier1" : "none")
if #available(macOS 15.0, *) {
    props["architectureName"] = device.architecture.name
}
if #available(macOS 14.0, *) {
    props["peerGroupID"] = device.peerGroupID
    props["peerIndex"] = device.peerIndex
    props["peerCount"] = device.peerCount
}

// MTLGPUFamily walk — what THIS device claims, family by family
var fams = [String: Bool]()
let famList: [(String, MTLGPUFamily)] = [
    ("apple1", .apple1), ("apple2", .apple2), ("apple3", .apple3), ("apple4", .apple4),
    ("apple5", .apple5), ("apple6", .apple6), ("apple7", .apple7), ("apple8", .apple8),
    ("apple9", .apple9), ("mac1", .mac1), ("mac2", .mac2),
    ("common1", .common1), ("common2", .common2), ("common3", .common3),
    ("metal3", .metal3), ("metal4", .metal4),
]
for (label, fam) in famList {
    fams[label] = device.supportsFamily(fam)
}
props["gpuFamilies"] = fams
report["device"] = props

// ---------- probe helpers ----------
let queue = device.makeCommandQueue()!

func runAndWait(_ cb: MTLCommandBuffer) -> Bool {
    let sem = DispatchSemaphore(value: 0)
    var ok = false
    cb.addCompletedHandler { c in ok = (c.status == .completed) ; sem.signal() }
    cb.commit()
    sem.wait()
    return ok
}

func makePS(_ src: String, lang: MTLLanguageVersion) -> (MTLComputePipelineState?, String) {
    let opt = MTLCompileOptions()
    opt.languageVersion = lang
    do {
        let lib = try device.makeLibrary(source: src, options: opt)
        let fn = lib.makeFunction(name: "k")!
        let ps = try device.makeComputePipelineState(function: fn)
        return (ps, "")
    } catch {
        let ns = error as NSError
        return (nil, ns.localizedDescription)
    }
}

var probes = [String: Any]()

// ---------- probe: trivial kernel → SIMD width + max threadgroup dispatch ----------
let trivialSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float *out [[buffer(0)]]) {
    out[0] = 1.0f;
}
"""
let (trivPS, trivErr) = makePS(trivialSrc, lang: .version3_1)
if let ps = trivPS {
    var p = [String: Any]()
    p["compiled"] = true
    p["threadExecutionWidth"] = ps.threadExecutionWidth
    p["pipelineMaxTotalThreadsPerThreadgroup"] = ps.maxTotalThreadsPerThreadgroup
    // dispatch one threadgroup at the DEVICE-reported max width to prove it
    let buf = device.makeBuffer(length: 16, options: .storageModeShared)!
    if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
        enc.setComputePipelineState(ps)
        enc.setBuffer(buf, offset: 0, index: 0)
        let mxW = device.maxThreadsPerThreadgroup.width
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: mxW, height: 1, depth: 1))
        enc.endEncoding()
        p["maxThreadsDispatchCompleted"] = runAndWait(cb)
        let v = buf.contents().bindMemory(to: Float.self, capacity: 1).pointee
        p["wroteValue"] = v == 1.0
    } else { p["maxThreadsDispatchCompleted"] = false }
    probes["trivial"] = p
} else {
    probes["trivial"] = ["compiled": false, "error": trivErr]
}

// ---------- probe: device atomic float add ----------
// 8 threadgroups x 32 lanes x 4096 adds, all onto one scalar. Exact when the
// accumulator stays an integer-valued float, which 1048576 is.
let atomicSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device atomic_float *acc [[buffer(0)]], constant uint *nit [[buffer(1)]]) {
    for (uint i = 0; i < *nit; ++i) {
        atomic_fetch_add_explicit(acc, 1.0f, memory_order_relaxed);
    }
}
"""
let (atomPS, atomErr) = makePS(atomicSrc, lang: .version3_1)
if let ps = atomPS {
    var p = [String: Any]()
    p["compiled"] = true
    let acc = device.makeBuffer(length: 16, options: .storageModeShared)!
    let nitv: UInt32 = 4096
    let nit = device.makeBuffer(bytes: [nitv], length: 4, options: .storageModeShared)!
    acc.contents().bindMemory(to: Float.self, capacity: 1).pointee = 0.0
    if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
        enc.setComputePipelineState(ps)
        enc.setBuffer(acc, offset: 0, index: 0)
        enc.setBuffer(nit, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        let ok = runAndWait(cb)
        let got = acc.contents().bindMemory(to: Float.self, capacity: 1).pointee
        p["executed"] = ok
        p["expected"] = Float(8 * 32 * nitv)
        p["got"] = got
        p["numericallyCorrect"] = got == Float(8 * 32 * nitv)
    } else { p["executed"] = false }
    probes["atomicFloatDeviceAdd"] = p
} else {
    probes["atomicFloatDeviceAdd"] = ["compiled": false, "error": atomErr]
}

// ---------- probe: simdgroup matrix 8x8x8 multiply at fp32 / f16 / bf16 ----------
func simdprobe(src: String, inBytes: Int,
               fill: (UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Void,
               check: (UnsafeMutableRawPointer) -> (Bool, [String: Any])) -> [String: Any] {
    let (ps, err) = makePS(src, lang: .version3_1)
    if ps == nil {
        return ["compiled": false, "error": err]
    }
    let a = device.makeBuffer(length: 8 * 8 * inBytes, options: .storageModeShared)!
    let b = device.makeBuffer(length: 8 * 8 * inBytes, options: .storageModeShared)!
    let c = device.makeBuffer(length: 8 * 8 * 4, options: .storageModeShared)!
    fill(a.contents(), b.contents())
    var p = [String: Any]()
    p["compiled"] = true
    if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
        enc.setComputePipelineState(ps!)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(c, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        let ok = runAndWait(cb)
        p["executed"] = ok
        let (good, extra) = check(c.contents())
        p["numericallyCorrect"] = good
        p["check"] = extra
    } else { p["executed"] = false }
    return p
}

// inputs chosen exactly representable so any correct matmul must be exact
var aref = [Float](repeating: 0, count: 64)
var bref = [Float](repeating: 0, count: 64)
for i in 0..<8 { for j in 0..<8 { aref[i*8+j] = Float((i % 3) + 1); bref[i*8+j] = Float((j % 4) - 1) } }
let f32Src = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float *A [[buffer(0)]], device float *B [[buffer(1)]], device float *C [[buffer(2)]]) {
    simdgroup_matrix<float, 8, 8> a;
    simdgroup_load(a, A, 8, ulong2(0, 0));
    simdgroup_matrix<float, 8, 8> b;
    simdgroup_load(b, B, 8, ulong2(0, 0));
    simdgroup_matrix<float, 8, 8> c;
    simdgroup_multiply(c, a, b);
    simdgroup_store(c, C, 8, ulong2(0, 0));
}
"""
probes["simdgroupMatrixF32_8x8x8"] = simdprobe(src: f32Src, inBytes: 4,
    fill: { ap, bp in
        let av = ap.bindMemory(to: Float.self, capacity: 64); for i in 0..<64 { av[i] = aref[i] }
        let bv = bp.bindMemory(to: Float.self, capacity: 64); for i in 0..<64 { bv[i] = bref[i] }
    },
    check: { cp in
        let cv = cp.bindMemory(to: Float.self, capacity: 64)
        var ok = true; var bad = 0
        for i in 0..<8 { for j in 0..<8 {
            var s: Float = 0
            for t in 0..<8 { s += aref[i*8+t] * bref[t*8+j] }
            if cv[i*8+j] != s { ok = false; bad += 1 }
        } }
        return (ok, ["mismatches": bad])
    })

let f16Src = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device half *A [[buffer(0)]], device half *B [[buffer(1)]], device float *C [[buffer(2)]]) {
    simdgroup_matrix<half, 8, 8> a;
    simdgroup_load(a, A, 8, ulong2(0, 0));
    simdgroup_matrix<half, 8, 8> b;
    simdgroup_load(b, B, 8, ulong2(0, 0));
    simdgroup_matrix<float, 8, 8> c;
    simdgroup_multiply(c, a, b);
    simdgroup_store(c, C, 8, ulong2(0, 0));
}
"""
let hvals: [UInt16] = [0x3800, 0x3C00, 0x4000, 0x4200, 0x3F00, 0x4400, 0x3400, 0x4800] // 0.5 1 2 3 0.5 4 0.25 8
func hbits(_ v: UInt16) -> Float { Float(UInt32(v) << 16) }
probes["simdgroupMatrixF16_8x8x8"] = simdprobe(src: f16Src, inBytes: 2,
    fill: { ap, bp in
        let av = ap.bindMemory(to: UInt16.self, capacity: 64); for i in 0..<64 { av[i] = hvals[i % 8] }
        let bv = bp.bindMemory(to: UInt16.self, capacity: 64); for i in 0..<64 { bv[i] = hvals[(i / 8) % 8] }
    },
    check: { cp in
        let cv = cp.bindMemory(to: Float.self, capacity: 64)
        var ok = true; var bad = 0
        for i in 0..<8 { for j in 0..<8 {
            var s: Float = 0
            for t in 0..<8 { s += hbits(hvals[(i*8+t) % 8]) * hbits(hvals[((t*8+j) / 8) % 8]) }
            if cv[i*8+j] != s { ok = false; bad += 1 }
        } }
        return (ok, ["mismatches": bad])
    })

let bf16Src = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device bfloat *A [[buffer(0)]], device bfloat *B [[buffer(1)]], device float *C [[buffer(2)]]) {
    simdgroup_matrix<bfloat, 8, 8> a;
    simdgroup_load(a, A, 8, ulong2(0, 0));
    simdgroup_matrix<bfloat, 8, 8> b;
    simdgroup_load(b, B, 8, ulong2(0, 0));
    simdgroup_matrix<float, 8, 8> c;
    simdgroup_multiply(c, a, b);
    simdgroup_store(c, C, 8, ulong2(0, 0));
}
"""
let bfvals: [UInt16] = [0x3F80, 0x4000, 0x3F00, 0x4040, 0x3FC0, 0x4100, 0x3E80, 0x4180] // 1 2 0.5 3 1.5 8 0.25 16
func bfbits(_ v: UInt16) -> Float { Float(UInt32(v) << 16) }
probes["simdgroupMatrixBF16_8x8x8"] = simdprobe(src: bf16Src, inBytes: 2,
    fill: { ap, bp in
        let av = ap.bindMemory(to: UInt16.self, capacity: 64); for i in 0..<64 { av[i] = bfvals[i % 8] }
        let bv = bp.bindMemory(to: UInt16.self, capacity: 64); for i in 0..<64 { bv[i] = bfvals[(i / 8) % 8] }
    },
    check: { cp in
        let cv = cp.bindMemory(to: Float.self, capacity: 64)
        var ok = true; var bad = 0
        for i in 0..<8 { for j in 0..<8 {
            var s: Float = 0
            for t in 0..<8 { s += bfbits(bfvals[(i*8+t) % 8]) * bfbits(bfvals[((t*8+j) / 8) % 8]) }
            if cv[i*8+j] != s { ok = false; bad += 1 }
        } }
        return (ok, ["mismatches": bad])
    })

// ---------- probe: dynamic threadgroup memory at the reported max ----------
// Every lane writes the same sentinel bytes, barrier, every lane reads them
// back. Same-value writes keep it race-benign while proving the full reported
// threadgroup allocation is backed.
let tgMax = device.maxThreadgroupMemoryLength
let tgSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void k(threadgroup char *tg [[threadgroup(0)]], device char *out [[buffer(0)]]) {
    tg[0] = 0x2A;
    tg[\(tgMax) - 1] = 0x2B;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[0] = tg[0] + tg[\(tgMax) - 1];
}
"""
let (tgPS, tgErr) = makePS(tgSrc, lang: .version3_1)
if let ps = tgPS {
    var p = [String: Any]()
    p["compiled"] = true
    let out = device.makeBuffer(length: 16, options: .storageModeShared)!
    if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
        enc.setComputePipelineState(ps)
        enc.setThreadgroupMemoryLength(tgMax, index: 0)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        p["executed"] = runAndWait(cb)
        let v = out.contents().bindMemory(to: UInt8.self, capacity: 16)
        p["sumOK"] = v[0] == (0x2A + 0x2B)
    } else { p["executed"] = false }
    probes["threadgroupMemoryAtMax"] = p
} else {
    probes["threadgroupMemoryAtMax"] = ["compiled": false, "error": tgErr]
}

report["probes"] = probes

let data = try! JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
print(String(data: data, encoding: .utf8)!)
