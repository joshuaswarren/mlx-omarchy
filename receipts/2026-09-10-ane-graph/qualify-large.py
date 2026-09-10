#!/usr/bin/env python3
import argparse
import ctypes
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

ENV_FLAG = "MLX_OMARCHY_ANE_REGION"
ATOL = 0.02
RTOL = 0.02


def import_runtime(root):
    sys.path[:0] = [os.fspath(root), os.fspath(Path(root) / "tools")]
    from research import inspect_anec
    import h13_run_linux
    return inspect_anec, h13_run_linux


def stop(process):
    process.terminate()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)


def receive(connection, process, timeout):
    if not connection.poll(timeout):
        stop(process)
        raise TimeoutError(f"ANE worker exceeded {timeout:.3f}s deadline")
    reply = connection.recv()
    if reply.get("status") == "error":
        raise RuntimeError(reply["detail"] + "\n" + reply["traceback"])
    return reply


def refill_dynamic(prepared, buffers, constant_indexes):
    for index, data in enumerate(buffers):
        if index in constant_indexes:
            continue
        target = prepared.input_buffers[index]
        if len(data) != len(target):
            raise RuntimeError("dynamic surface changed allocation size")
        ctypes.memmove(target, data, len(data))


def worker_main(connection, compiler_root, package, libane, input_name, input_shm,
                output_name, output_shm):
    input_region = shared_memory.SharedMemory(name=input_shm)
    output_region = shared_memory.SharedMemory(name=output_shm)
    records = []
    try:
        inspect_anec, runtime = import_runtime(compiler_root)
        manifest, _ = inspect_anec.load_package(package)
        tensors = manifest["tensors"]
        if manifest["target"] != "H13" or manifest["dispatchPlan"] != list(
            range(len(manifest["programs"]))
        ):
            raise ValueError("large region requires an ordered H13 package")
        setup_started = time.perf_counter_ns()
        constant_pack_ns = 0
        prepared_ns = 0
        adapter = runtime.LibANEAdapter(libane)
        package_path = Path(package).resolve()
        for program_index in manifest["dispatchPlan"]:
            program = manifest["programs"][program_index]
            placeholders = []
            constant_indexes = set()
            for index, binding in enumerate(program["inputs"]):
                if binding.get("binding") == "constant":
                    started = time.perf_counter_ns()
                    placeholders.append(runtime._constant_buffer(program, binding))
                    constant_pack_ns += time.perf_counter_ns() - started
                    constant_indexes.add(index)
                else:
                    placeholders.append(bytes(binding["allocationBytes"]))
            anec = inspect_anec.local_file(
                package_path, program["file"], program["bytes"]
            )
            started = time.perf_counter_ns()
            prepared = runtime.PreparedProgram(
                adapter,
                anec,
                placeholders,
                [binding["allocationBytes"] for binding in program["outputs"]],
            )
            prepared_ns += time.perf_counter_ns() - started
            records.append((program_index, program, prepared, constant_indexes))
        setup_finished = time.perf_counter_ns()
        identity = {
            "pid": os.getpid(),
            "programs": len(records),
            "handles": [int(record[2].handle) for record in records],
            "constant_pack_once_us": constant_pack_ns / 1000,
            "prepared_programs_once_us": prepared_ns / 1000,
            "worker_setup_total_us": (setup_finished - setup_started) / 1000,
        }
        connection.send({"status": "ready", "identity": identity})
        expected_sequence = 0
        while True:
            request = connection.recv()
            received_ns = time.perf_counter_ns()
            if request["command"] == "shutdown":
                connection.send({"status": "stopping", "sequence": expected_sequence})
                break
            sequence = request["sequence"]
            if request["command"] != "run" or sequence != expected_sequence:
                raise RuntimeError(
                    f"stream order violation: expected {expected_sequence}, got {sequence}"
                )
            dense_inputs = {input_name: bytes(input_region.buf[: tensors[input_name]["logicalBytes"]])}
            intermediate_regions = {}
            output_regions = {}
            pack_ns = transfer_ns = exec_ns = read_ns = 0
            for program_index, program, prepared, constant_indexes in records:
                started = time.perf_counter_ns()
                buffers = []
                for binding in program["inputs"]:
                    name = binding["name"]
                    if binding.get("binding") == "constant":
                        buffers.append(None)
                    elif tensors[name]["role"] == "intermediate":
                        buffers.append(runtime._intermediate_buffer(
                            binding, tensors, intermediate_regions.get(name, [])
                        ))
                    else:
                        buffers.append(runtime._runtime_buffer(
                            dense_inputs[name], binding, tensors
                        ))
                refill_dynamic(prepared, buffers, constant_indexes)
                pack_ns += time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                prepared.transfer_in()
                transfer_ns += time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                prepared.submit()
                exec_ns += time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                data = prepared.transfer_out()
                read_ns += time.perf_counter_ns() - started
                runtime._collect_outputs(
                    tensors, program, data, intermediate_regions, output_regions
                )
            started = time.perf_counter_ns()
            dense_output = runtime._unpack_outputs(
                manifest, output_regions, {output_name}
            )[output_name]
            output_region.buf[: len(dense_output)] = dense_output
            unpack_ns = time.perf_counter_ns() - started
            send_ns = time.perf_counter_ns()
            expected_sequence += 1
            connection.send({
                "status": "ok",
                "sequence": sequence,
                "identity": identity,
                "received_ns": received_ns,
                "send_ns": send_ns,
                "worker_total_ns": send_ns - received_ns,
                "pack_ns": pack_ns,
                "device_input_ns": transfer_ns,
                "ane_exec_ns": exec_ns,
                "device_output_ns": read_ns,
                "unpack_ns": unpack_ns,
            })
    except BaseException as error:
        try:
            connection.send({
                "status": "error",
                "detail": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
        except BaseException:
            pass
    finally:
        for _, _, prepared, _ in records:
            try:
                prepared.close()
            except BaseException:
                pass
        input_region.close()
        output_region.close()
        connection.close()


def gpu_once(mx, compiled, x, weight):
    started = time.perf_counter_ns()
    result = compiled(x, weight)
    mx.eval(result)
    return result, (time.perf_counter_ns() - started) / 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-root", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--libane", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if os.environ.get(ENV_FLAG) != "1":
        parser.error(f"set {ENV_FLAG}=1 to run the off-default ANE qualification path")

    inspect_anec, _ = import_runtime(args.compiler_root)
    manifest, _ = inspect_anec.load_package(args.package)
    tensors = manifest["tensors"]
    inputs = [name for name, tensor in tensors.items() if tensor["role"] == "input"]
    outputs = [name for name, tensor in tensors.items() if tensor["role"] == "output"]
    if inputs != ["x"] or outputs != ["y"]:
        raise ValueError(f"expected x -> y package, got {inputs} -> {outputs}")
    input_bytes = tensors["x"]["logicalBytes"]
    output_bytes = tensors["y"]["logicalBytes"]

    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    device_info = mx.device_info()
    identity = {
        "mlx_version": mx.__version__,
        "cooperative_matrix_f32_8": device_info["cooperative_matrix_f32_8"],
        "device_info": device_info,
    }
    print(json.dumps({"benchmark_identity": identity}, sort_keys=True), flush=True)

    x_host = np.frombuffer(args.input.read_bytes(), dtype="<f2").reshape(1, 896)
    weight_file = args.weights.read_bytes()
    weight_host = np.frombuffer(
        weight_file, dtype="<f2", count=4864 * 896, offset=64
    ).reshape(4864, 896)
    retained_expected = np.frombuffer(args.expected.read_bytes(), dtype="<f2").reshape(1, 4864)
    x = mx.array(x_host, dtype=mx.float16)
    weight = mx.array(weight_host, dtype=mx.float16)
    mx.eval(x, weight)
    compiled = mx.compile(lambda value, matrix: value @ matrix.T)
    reference, _ = gpu_once(mx, compiled, x, weight)
    gpu_reference = np.asarray(reference).astype(np.float32)
    compiler_reference = retained_expected.astype(np.float32)
    gpu_difference = np.abs(gpu_reference - compiler_reference)
    gpu_allowed = ATOL + RTOL * np.abs(compiler_reference)
    gpu_numerical_passed = bool(np.all(gpu_difference <= gpu_allowed))

    input_region = shared_memory.SharedMemory(create=True, size=input_bytes)
    output_region = shared_memory.SharedMemory(create=True, size=output_bytes)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=worker_main,
        args=(child, os.fspath(args.compiler_root), os.fspath(args.package),
              os.fspath(args.libane), "x", input_region.name, "y", output_region.name),
    )
    started = time.perf_counter_ns()
    process.start()
    child.close()
    samples = []
    gpu_samples = []
    numerical = []
    try:
        ready = receive(parent, process, args.timeout)
        ready["identity"]["parent_observed_setup_us"] = (time.perf_counter_ns() - started) / 1000
        for sequence in range(args.warmup + args.iterations):
            gpu_result, gpu_us = gpu_once(mx, compiled, x, weight)
            total_started = time.perf_counter_ns()
            dense = np.ascontiguousarray(np.asarray(x), dtype="<f2").tobytes()
            input_region.buf[:input_bytes] = dense
            sent_ns = time.perf_counter_ns()
            parent.send({"command": "run", "sequence": sequence})
            response = receive(parent, process, args.timeout)
            received_ns = time.perf_counter_ns()
            result_host = np.frombuffer(bytes(output_region.buf[:output_bytes]), dtype="<f2").reshape(1, 4864)
            result = mx.array(result_host, dtype=mx.float16)
            mx.eval(result)
            finished_ns = time.perf_counter_ns()
            actual = np.asarray(result).astype(np.float32)
            difference = np.abs(actual - compiler_reference)
            allowed = ATOL + RTOL * np.abs(compiler_reference)
            passed = bool(np.all(difference <= allowed))
            numerical.append({
                "passed": passed,
                "max_abs": float(difference.max()),
                "max_allowed": float(allowed.max()),
            })
            if not passed:
                raise RuntimeError(f"ANE output failed compiler-oracle tolerance at sequence {sequence}")
            if sequence >= args.warmup:
                gpu_samples.append(gpu_us)
                metric = {
                    "input_copy": (sent_ns - total_started) / 1000,
                    "request_ipc": (response["received_ns"] - sent_ns) / 1000,
                    "worker_pack": response["pack_ns"] / 1000,
                    "worker_device_input": response["device_input_ns"] / 1000,
                    "worker_ane_exec": response["ane_exec_ns"] / 1000,
                    "worker_device_output": response["device_output_ns"] / 1000,
                    "worker_unpack": response["unpack_ns"] / 1000,
                    "worker_total": response["worker_total_ns"] / 1000,
                    "response_ipc": (received_ns - response["send_ns"]) / 1000,
                    "output_copy": (finished_ns - received_ns) / 1000,
                    "total": (finished_ns - total_started) / 1000,
                }
                samples.append(metric)
        parent.send({"command": "shutdown"})
        stopped = receive(parent, process, args.timeout)
        process.join(args.timeout)
        if process.is_alive() or process.exitcode != 0:
            raise RuntimeError("large-region worker failed clean shutdown")
    finally:
        if process.is_alive():
            stop(process)
        parent.close()
        input_region.close()
        input_region.unlink()
        output_region.close()
        output_region.unlink()

    medians = {key: statistics.median(sample[key] for sample in samples) for key in samples[0]}
    gpu_median = statistics.median(gpu_samples)
    result = {
        "schema": "mlx-omarchy.ane-large-region-qualification.v1",
        "date": "2026-09-10",
        "region": "Qwen layer projection fp16 [1,896] @ [4864,896].T -> [1,4864]",
        "compiler_commit": subprocess.check_output(
            ["git", "-C", os.fspath(args.compiler_root), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "programs": len(manifest["programs"]),
        "task_descriptors": sum(
            program["taskDescriptors"] for program in manifest["programs"]
        ),
        "benchmark_identity": identity,
        "env_flag": ENV_FLAG,
        "default_path_enabled": False,
        "one_time": ready["identity"],
        "latency": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "gpu_median_us": gpu_median,
            "ane_median_us": medians,
            "samples": samples,
            "ane_to_gpu_ratio": medians["total"] / gpu_median,
            "passes_10_percent_bar": medians["total"] <= 0.9 * gpu_median,
        },
        "gpu_numerical": {
            "passed_against_compiler_oracle": gpu_numerical_passed,
            "max_abs": float(gpu_difference.max()),
            "atol": ATOL,
            "rtol": RTOL,
        },
        "numerical": {
            "passed": all(sample["passed"] for sample in numerical),
            "atol": ATOL,
            "rtol": RTOL,
            "samples": len(numerical),
            "max_abs": max(sample["max_abs"] for sample in numerical),
        },
        "shutdown": {"worker_exit_code": process.exitcode, "reply": stopped},
        "verdict": "keep" if medians["total"] <= 0.9 * gpu_median else "reject",
        "compiler_refusal": "896x896 two-runtime-operand fp16 matmul is outside the current decoded H13 parity envelope; this is the largest retained model-relevant supported projection input.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "mlx_version": mx.__version__,
        "cooperative_matrix_f32_8": identity["cooperative_matrix_f32_8"],
        "gpu_median_us": gpu_median,
        "ane_total_median_us": medians["total"],
        "ane_exec_median_us": medians["worker_ane_exec"],
        "worker_pack_median_us": medians["worker_pack"],
        "worker_unpack_median_us": medians["worker_unpack"],
        "passes_10_percent_bar": result["latency"]["passes_10_percent_bar"],
        "verdict": result["verdict"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
