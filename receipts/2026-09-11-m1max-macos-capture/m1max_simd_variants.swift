// m1max_simd_variants.swift — f16/bf16 simdgroup_matrix multiply variants
// Companion to m1max_metal_limits.swift. v2's mixed-precision (half in,
// float out) probes executed but mismatched; this probes same-type and
// mixed variants and dumps raw output bits for diagnosis.
import Metal
import Foundation

guard let device = MTLCreateSystemDefaultDevice() else {
    print("{\"error\": \"no default MTLDevice\"}")
    exit(1)
}
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
        return (nil, (error as NSError).localizedDescription)
    }
}

// A[r][c] = aval[c], B[r][c] = bval[r]; C[r][c] must equal sum_t aval[t]*bval[t]
let aval: [UInt16] = [0x3F80, 0x4000, 0x3F00, 0x4040] // 1 2 1.75 3 (bf16 bits)
let bval: [UInt16] = [0x3F80, 0x4000, 0x3F00, 0x4040]
func f32bits(_ v: UInt16) -> Float { Float(UInt32(v) << 16) }
var expect: Float = 0
for t in 0..<8 { expect += f32bits(aval[t % 4]) * f32bits(bval[t % 4]) }

func variantProbe(name: String, src: String, inTypeIs16bit: Bool,
                  outIsFloat32: Bool) -> [String: Any] {
    let (ps, err) = makePS(src, lang: .version3_1)
    let elem = inTypeIs16bit ? 2 : 4
    let a = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
    let b = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
    let c = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
    if inTypeIs16bit {
        let av = a.contents().bindMemory(to: UInt16.self, capacity: 64)
        let bv = b.contents().bindMemory(to: UInt16.self, capacity: 64)
        for i in 0..<64 { av[i] = aval[i % 4]; bv[i] = bval[(i / 8) % 4] }
    } else {
        let av = a.contents().bindMemory(to: Float.self, capacity: 64)
        let bv = b.contents().bindMemory(to: Float.self, capacity: 64)
        for i in 0..<64 { av[i] = f32bits(aval[i % 4]); bv[i] = f32bits(bval[(i / 8) % 4]) }
    }
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
        p["executed"] = runAndWait(cb)
        let raw = c.contents()
        var words = [UInt32]()
        for i in 0..<8 {
            words.append(raw.load(fromByteOffset: i * 4, as: UInt32.self))
        }
        p["first8cells_bits"] = words
        var bad = 0
        if outIsFloat32 {
            let cv = raw.bindMemory(to: Float.self, capacity: 64)
            for i in 0..<64 { if cv[i] != expect { bad += 1 } }
        } else {
            let cv = raw.bindMemory(to: UInt16.self, capacity: 64)
            for i in 0..<64 {
                let asF = Float(UInt32(cv[i]) << 16)
                if asF != expect { bad += 1 }
            }
        }
        p["mismatches_vs_expected"] = bad
        p["numericallyCorrect"] = bad == 0
    }
    return p
}

var out: [String: Any] = [:]

// half x half -> half (same type multiply, half accumulator)
let f16same = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device half *A [[buffer(0)]], device half *B [[buffer(1)]], device half *C [[buffer(2)]]) {
    simdgroup_matrix<half, 8, 8> a;
    simdgroup_load(a, A, 8, ulong2(0, 0));
    simdgroup_matrix<half, 8, 8> b;
    simdgroup_load(b, B, 8, ulong2(0, 0));
    simdgroup_matrix<half, 8, 8> c;
    simdgroup_multiply(c, a, b);
    simdgroup_store(c, C, 8, ulong2(0, 0));
}
"""
out["f16_same_half_out"] = variantProbe(name: "f16_same", src: f16same, inTypeIs16bit: true, outIsFloat32: false)

// half x half -> float (mixed accumulate)
let f16mixed = """
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
out["f16_mixed_float_out"] = variantProbe(name: "f16_mixed", src: f16mixed, inTypeIs16bit: true, outIsFloat32: true)

// bfloat x bfloat -> bfloat (same type)
let bf16same = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device bfloat *A [[buffer(0)]], device bfloat *B [[buffer(1)]], device bfloat *C [[buffer(2)]]) {
    simdgroup_matrix<bfloat, 8, 8> a;
    simdgroup_load(a, A, 8, ulong2(0, 0));
    simdgroup_matrix<bfloat, 8, 8> b;
    simdgroup_load(b, B, 8, ulong2(0, 0));
    simdgroup_matrix<bfloat, 8, 8> c;
    simdgroup_multiply(c, a, b);
    simdgroup_store(c, C, 8, ulong2(0, 0));
}
"""
out["bf16_same_bf_out"] = variantProbe(name: "bf16_same", src: bf16same, inTypeIs16bit: true, outIsFloat32: false)

// bfloat x bfloat -> float (mixed accumulate)
let bf16mixed = """
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
out["bf16_mixed_float_out"] = variantProbe(name: "bf16_mixed", src: bf16mixed, inTypeIs16bit: true, outIsFloat32: true)

// float x float -> float (control, must match v2 success)
let f32ctl = """
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
out["f32_control_float_out"] = variantProbe(name: "f32_ctl", src: f32ctl, inTypeIs16bit: false, outIsFloat32: true)

// NOTE: the two 16-bit-input variants store to a 32-bit-out buffer only in the
// mixed cases; the same-type cases store 16-bit elements. The first-8-cells
// dump reads 32-bit words in all cases for bit-level diagnosis.


// discriminator: all inputs constant 2.0 -> every C cell must be 32.0 exactly
// in every dtype. Separates layout/indexing errors from multiply semantics.
func scalarVariant(name: String, tname: String, in16: Bool) -> [String: Any] {
    let src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void k(device \(tname) *A [[buffer(0)]], device \(tname) *B [[buffer(1)]], device \(tname) *C [[buffer(2)]]) {
        simdgroup_matrix<\(tname), 8, 8> a;
        simdgroup_load(a, A, 8, ulong2(0, 0));
        simdgroup_matrix<\(tname), 8, 8> b;
        simdgroup_load(b, B, 8, ulong2(0, 0));
        simdgroup_matrix<\(tname), 8, 8> c;
        simdgroup_multiply(c, a, b);
        simdgroup_store(c, C, 8, ulong2(0, 0));
    }
    """
    let (ps, err) = makePS(src, lang: .version3_1)
    if ps == nil { return ["compiled": false, "error": err] }
    let a = device.makeBuffer(length: 256, options: .storageModeShared)!
    let b = device.makeBuffer(length: 256, options: .storageModeShared)!
    let c = device.makeBuffer(length: 256, options: .storageModeShared)!
    if in16 {
        let av = a.contents().bindMemory(to: UInt16.self, capacity: 64)
        let bv = b.contents().bindMemory(to: UInt16.self, capacity: 64)
        for i in 0..<64 { av[i] = 0x4000; bv[i] = 0x4000 } // 2.0 in half AND bf16
    } else {
        let av = a.contents().bindMemory(to: Float.self, capacity: 64)
        let bv = b.contents().bindMemory(to: Float.self, capacity: 64)
        for i in 0..<64 { av[i] = 2.0; bv[i] = 2.0 }
    }
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
        p["executed"] = runAndWait(cb)
        var vals = [Any]()
        if in16 {
            let cv = c.contents().bindMemory(to: UInt16.self, capacity: 64)
            let f0 = Float(UInt32(cv[0]) << 16)
            var uniform = true
            for i in 1..<64 { if cv[i] != cv[0] { uniform = false } }
            vals.append(f0)
            p["uniform"] = uniform
            p["matches32"] = f0 == 32.0
        } else {
            let cv = c.contents().bindMemory(to: Float.self, capacity: 64)
            var uniform = true
            for i in 1..<64 { if cv[i] != cv[0] { uniform = false } }
            vals.append(cv[0])
            p["uniform"] = uniform
            p["matches32"] = cv[0] == 32.0
        }
        p["cell0"] = vals.first ?? "n/a"
    }
    return p
}
out["discriminator_f32_scalar2"] = scalarVariant(name: "f32s", tname: "float", in16: false)
out["discriminator_f16_scalar2"] = scalarVariant(name: "f16s", tname: "half", in16: true)
out["discriminator_bf16_scalar2"] = scalarVariant(name: "bf16s", tname: "bfloat", in16: true)

let data = try! JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys])
print(String(data: data, encoding: .utf8)!)
