// m1max_blit_roof.swift — GPU large-copy roof, comparable instrument to
// receipts/2026-09-11-q4-memory-roof (large streaming copy, 256 MB, wall-anchored).
// MTLBlitCommandEncoder copy from 256 MB buffer to 256 MB buffer; new command
// buffer + wait per rep; wall time via clock_gettime_nsec_np(CLOCK_MONOTONIC).
import Metal
import Foundation

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    print("{\"error\": \"no device\"}")
    exit(1)
}
let size = 256 * 1024 * 1024
let src = device.makeBuffer(length: size, options: .storageModeShared)!
let dst = device.makeBuffer(length: size, options: .storageModeShared)!
memset(src.contents(), 0x5A, size)

func oneCopy() -> Double {
    let t0 = clock_gettime_nsec_np(CLOCK_MONOTONIC)
    guard let cb = queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else { return -1 }
    blit.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: size)
    blit.endEncoding()
    let sem = DispatchSemaphore(value: 0)
    cb.addCompletedHandler { _ in sem.signal() }
    cb.commit()
    sem.wait()
    return Double(clock_gettime_nsec_np(CLOCK_MONOTONIC) - t0) / 1e9
}

// warmup x3
for _ in 0..<3 { _ = oneCopy() }
var samples: [Double] = []
for _ in 0..<20 {
    let t = oneCopy()
    if t > 0 { samples.append(t) }
}
let moved = Double(size) * 2.0 // read + write
let gbps = samples.map { moved / $0 / 1e9 }
let sorted = gbps.sorted()
let med = sorted[sorted.count / 2]
let out: [String: Any] = [
    "method": "MTLBlitCommandEncoder copy, 256 MB -> 256 MB, storageModeShared, wall-anchored CLOCK_MONOTONIC, 1 command buffer per rep, 3 warmup + 20 timed",
    "bytes_moved_per_rep": moved,
    "median_gbps": med,
    "min_gbps": sorted.first!,
    "max_gbps": sorted.last!,
    "all_gbps": gbps,
    "check_ok": memcmp(src.contents(), dst.contents(), size) == 0,
]
let data = try! JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys])
print(String(data: data, encoding: .utf8)!)
