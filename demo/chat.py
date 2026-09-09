"""Chat with a small local LLM on the Apple GPU through mlx-omarchy.

Installed by install.sh as `mlx-omarchy-demo`. First run downloads the model
(about 300 MB for the default Qwen2.5-0.5B-Instruct-4bit) into
~/.cache/huggingface.

    mlx-omarchy-demo                      # scripted first turn, then interactive chat
    mlx-omarchy-demo --once               # scripted turn only, then exit
    mlx-omarchy-demo --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
"""

import argparse
import sys

import mlx.core as mx
from mlx_lm import load, stream_generate

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
FIRST_PROMPT = "In three sentences, explain what makes Apple Silicon unusual for running language models on Linux."


def banner():
    info = mx.device_info()
    print(f"mlx-omarchy {mx.__version__}")
    print(f"device: {info.get('device_name', info)}")
    print()


def turn(model, tokenizer, messages, max_tokens):
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    reply = []
    last = None
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        print(response.text, end="", flush=True)
        reply.append(response.text)
        last = response
    print()
    if last is not None:
        print(
            f"  [prompt {last.prompt_tokens} tok @ {last.prompt_tps:.1f} tok/s, "
            f"generated {last.generation_tokens} tok @ {last.generation_tps:.1f} tok/s]"
        )
    return "".join(reply)


def main():
    parser = argparse.ArgumentParser(description="Chat with a local LLM on the Apple GPU through mlx-omarchy.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt", default=FIRST_PROMPT, help="scripted first message")
    parser.add_argument("--once", action="store_true", help="exit after the scripted first message")
    args = parser.parse_args()

    banner()
    print(f"loading {args.model} ...", flush=True)
    model, tokenizer = load(args.model)
    messages = []

    print(f"\nyou> {args.prompt}\nassistant> ", end="", flush=True)
    messages.append({"role": "user", "content": args.prompt})
    messages.append({"role": "assistant", "content": turn(model, tokenizer, messages, args.max_tokens)})
    if args.once:
        return

    print("\nInteractive chat. Empty line or Ctrl-D exits.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            print()
            return
        if not text:
            return
        messages.append({"role": "user", "content": text})
        print("assistant> ", end="", flush=True)
        messages.append({"role": "assistant", "content": turn(model, tokenizer, messages, args.max_tokens)})


if __name__ == "__main__":
    sys.exit(main())
