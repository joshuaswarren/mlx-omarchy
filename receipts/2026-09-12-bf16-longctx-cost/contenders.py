#!/usr/bin/env python3
import argparse
import os
import pathlib


SERVICES = {"llama-server", "llama-cli", "ollama", "omlx"}


def is_contender(pid, comm, argv, self_pid):
    if pid == self_pid:
        return False
    comm = comm.lower()
    if comm in SERVICES:
        return True
    if not comm.startswith("python") or len(argv) < 2:
        return False
    entry = pathlib.Path(argv[1]).name.lower()
    if entry.startswith("bench") or entry in {"generate.py", "server.py"}:
        return True
    return len(argv) >= 3 and argv[1] == "-m" and argv[2].lower().startswith("mlx")


def scan():
    hits = []
    self_pid = os.getpid()
    for proc in pathlib.Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            comm = (proc / "comm").read_text().strip()
            argv = [value.decode(errors="replace") for value in
                    (proc / "cmdline").read_bytes().split(b"\0") if value]
        except OSError:
            continue
        if is_contender(int(proc.name), comm, argv, self_pid):
            hits.append(f"{proc.name} {comm} {' '.join(argv)}")
    return hits


def self_test():
    self_pid = 42
    assert not is_contender(
        1, "bash", ["bash", "-c", 'pgrep -f "mlx-attention/probe.py"'], self_pid)
    assert not is_contender(
        self_pid, "python", ["/src/mlx/.work/venv/bin/python", "contenders.py"], self_pid)
    assert not is_contender(
        2, "python3", ["python3", "worker_watch.py", "--config", "/tmp/mlx.json"], self_pid)
    assert is_contender(
        3, "python", ["python", "/tmp/bench_decode.py", "--model", "qwen"], self_pid)
    assert is_contender(4, "python3.14", ["python3.14", "-m", "mlx_lm", "generate"], self_pid)
    assert is_contender(5, "llama-server", ["llama-server", "-m", "model.gguf"], self_pid)
    assert not is_contender(6, "python", ["python", "unrelated_worker.py"], self_pid)
    print("CONTENDER_CLASSIFIER_SELF_TEST_OK")


parser = argparse.ArgumentParser()
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()
if args.self_test:
    self_test()
else:
    found = scan()
    print("\n".join(found))
    raise SystemExit(bool(found))
