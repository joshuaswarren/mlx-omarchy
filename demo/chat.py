"""Chat with a user-chosen local LLM on the Apple GPU through mlx-omarchy.

Installed by install.sh as `mlx-omarchy-demo`. The Hugging Face model id
is required: pass `--model`, or (on an interactive terminal) the demo
asks for one before importing mlx or downloading anything. No model is
suggested or pinned — this checkpoint's `mlx_lm` path is unqualified for
the current generation; the verified current-generation path
(`mlx-community/Qwen3.8-27B-4bit` via `mlx-vlm`) is documented in the
top-level README Quick start. Run the demo only with a model you have
qualified on this stack; the first run downloads your chosen model into
~/.cache/huggingface.

    mlx-omarchy-demo --model MODEL_ID   # scripted first turn, then interactive chat
    mlx-omarchy-demo --model MODEL_ID --once
    mlx-omarchy-demo                    # asks for MODEL_ID on an interactive terminal
"""

import argparse
import sys


def resolve_model(argv=None):
    """Parse argv and return args with `model` resolved.

    Runs before any mlx/mlx_lm import so a missing or empty model id never
    triggers a download. Missing --model is prompted for on an interactive
    terminal and rejected elsewhere.
    """
    parser = argparse.ArgumentParser(description="Chat with a local LLM on the Apple GPU through mlx-omarchy.")
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        default=None,
        help="Hugging Face repo id of a model you have qualified on this stack (required; prompted for on an interactive terminal)",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt", default=FIRST_PROMPT, help="scripted first message")
    parser.add_argument("--once", action="store_true", help="exit after the scripted first message")
    args = parser.parse_args(argv)

    if args.model is not None and not args.model.strip():
        parser.error("--model must be a non-empty model id")
    if args.model is None:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            parser.error("--model is required when stdin/stdout is not an interactive terminal")
        try:
            answer = input("Hugging Face model id to chat with: ").strip()
        except EOFError:
            parser.error("--model is required (no model id entered)")
        if not answer:
            parser.error("--model is required (no model id entered)")
        args.model = answer
    return args


FIRST_PROMPT = "In three sentences, explain what makes Apple Silicon unusual for running language models on Linux."


def banner():
    import mlx.core as mx

    info = mx.device_info()
    print(f"mlx-omarchy {mx.__version__}")
    print(f"device: {info.get('device_name', info)}")
    print()


def turn(model, tokenizer, messages, max_tokens):
    from mlx_lm import stream_generate

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
    args = resolve_model()

    from mlx_lm import load

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
