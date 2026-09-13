#!/usr/bin/env python3
import argparse
import pathlib


def is_contender(comm, argv):
    comm = comm.lower()
    argv = argv.lower()
    return comm in {"llama-server", "llama-cli", "ollama", "omlx"} or (
        comm.startswith("python") and any(x in argv for x in
                                           ("bench", "generate", "mlx")))


def scan():
    hits = []
    for proc in pathlib.Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            comm = (proc / "comm").read_text().strip()
            argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace")
        except OSError:
            continue
        if is_contender(comm, argv):
            hits.append(f"{proc.name} {comm} {argv}")
    return hits


def self_test():
    assert not is_contender(
        "bash", 'bash -c while pgrep -f "mlx-attention/probe.py"; do sleep 5; done')
    assert is_contender("python", "python /tmp/bench_decode.py --model qwen")
    assert is_contender("python3.14", "python3.14 -m mlx_lm generate")
    assert is_contender("llama-server", "llama-server -m model.gguf")
    assert not is_contender("python", "python unrelated_worker.py")
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
