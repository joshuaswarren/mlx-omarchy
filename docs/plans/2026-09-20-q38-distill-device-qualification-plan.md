# Qwen3.8-35B-A3B-Distill Q4/G64 device qualification plan

Status: FROZEN test vector, awaiting a device window (t6001-test-host preferred;
M2-GPU-interleave only if the boot owner lends a slot that fits
transfer + load). No HF publication, no public catalog entry; the
artifact is local-only by decision of the owner (2026-09-20).

## Artifact under test (local, immutable)

Path on conversion host: $HOME/models/qwen3.8-35b-a3b-distill-q4-g64
Source: empero-ai/Qwen3.8-35B-A3B-Distill @ bcc2dbe2f21b213625df2dc1a5a690212373af07
Quant: affine 4-bit / group 64, routers mlp.gate + shared_expert_gate
at 8-bit / group 64 (per-path overrides in config.json), mode affine.
License apache-2.0 (source and output).

Frozen shard digests (sha256, verified twice on the conversion host;
re-verify on the device BEFORE any run and abort on mismatch):

| shard | sha256 |
|---|---|
| model-1-of-6.safetensors | 640d25f27ffbf25a9daa5bc8db147f23edc914e1f5303f99e381af25a0ef28af |
| model-2-of-6.safetensors | fc10593d6feafc4ebc68e4d3767e94532e1a1d5374827c8f009e67b691b693ee |
| model-3-of-6.safetensors | a762747f333bd1f89b8cab326b2911c1a66094668fbf908fa4a991d393341461 |
| model-4-of-6.safetensors | ca8a62e28c6132f1fbaf5a47c7cff8bd3e3879720897bfed1d0b6fa42615e22e |
| model-5-of-6.safetensors | ef843bd297584ac2eaac01fe12e6186a0fb6ddc2d0a4adc6d7afcc8bff7e1ce1 |
| model-6-of-6.safetensors | 413a4e3e34ad1e17c245732ad3cbba8752944a98763584caf3e032ead8fce4af |

config.json / tokenizer files must come from the same directory; the
plan pins the directory, not an HF download.

## Frozen test vector

Exact prompt string: `The capital of France is`

Tokenization is performed by the artifact's own tokenizer
(tokenizer.json, vocab.json, merges.txt from the pinned revision).
CPU reference encoding (mlx-lm 0.31.3, transformers 5.17.0 fast
tokenizer): record the encoded IDs in the device receipt; the device
must tokenize to the same IDs with the artifact's tokenizer — assert
this before generating.

CPU reference generation (greedy, mlx-lm 0.31.3, 4 CPU cores,
RSS peak 18.62 GiB), 16 tokens:

```
prompt:  "The capital of France is"
ids:     [11751, 11, 264, 3177, 3750, 364, 1141, 8807, 3712,
          11, 1880, 11, 321, 7431, 13, 1049]
text:    " Paris, a city known for its rich history, art, and culture. It"
```

Receipt: forward_receipt.json in the artifact directory (admission,
hashes, logits shape (1, 5, 248320), determinism across two runs).
Evidence log: full-forward2.log; fixture-level loader-contract tests:
tools/test_qwen38_distill_quant.py (16 tests, real mlx_lm module tree).

### Pass rule (frozen before any device run)

1. Tokenizer gate: device-side encoded prompt IDs == CPU-encoded IDs.
2. Greedy generation, temperature 0, no sampler overrides,
   `max_tokens=16`, stop at EOS.
3. PASS: all 16 generated IDs equal the CPU reference sequence,
   and a second device run reproduces the same sequence
   (device determinism).
4. DIVERGENCE: any differing ID. Record the full sequence and the
   first differing position with both logits-space top-5. Divergence
   does NOT auto-fail the artifact — quantized GPU and CPU kernels
   may differ in low-order bits — but the qualification is NOT
   granted on divergence; the trace goes to the owner for review.
5. The first generated ID being anything other than 11751
   (` Paris`) additionally triggers immediate divergence analysis
   before any further tokens are compared.

No BF16-parity claim is made or derivable from this protocol; the CPU
run is the locked reference fixture per the repo's comparison rule.

## Managed-memory budget (for the device admission system)

| component | bytes |
|---|---|
| packed weights (text-resident) | 19,537,340,176 |
| fixed GDN state (ssm 60.0 MiB fp32 + conv 1.4 MiB) | 64,440,320 |
| KV cache | 20,480 B/token (10 full-attn layers, 2 kv heads x 256, K+V bf16) |
| workspace margin | device-owned; measured peak overrides (ServeCatalog budget API) |

Reservation floor: 19,601,780,496 B resident from step one
(weights + fixed state). The 16-token contract adds ~0.33 MiB KV.
Device admission must use actual MemAvailable/cgroup headroom, per the
CPU-run precedent.

## Device queue and host preference

1. t6001-test-host (preferred): persistent disk — the 19G artifact transfers
   once and survives reboots; serves later Bonsai/Laya/SPA-queued
   windows without re-transfer.
2. M2 GPU-interleave source gap: only if the boot owner lends a slot
   where transfer + load provably fit the memory budget and the
   transfer survives a reset cycle (repeated M2 resets cost repeated
   19G transfers — this is why t6001-test-host is preferred).

Transfer integrity: sha256 the six shards after copy on the device
host; abort on any mismatch.

## Run protocol on the device

Stage G (GPU CLI): `mlx_lm.generate --model <artifact-dir> --prompt
"The capital of France is" --max-tokens 16 --temp 0` — record raw IDs.
Note: mlx_lm.generate_step takes a 1-D [S] prompt tensor (model()
takes [B, S]) — use the CLI or handle shapes explicitly.

Stage H (HTTP): `python -m mlx_lm.server --model <artifact-dir>
--host 127.0.0.1 --port 8080`; send the same prompt via a chat
request at temperature 0; record the generated text and IDs. Text-
level comparison only (the server applies its own chat template and
sampling defaults must be pinned to temperature 0 / greedy).

Both stages: two runs each; receipts include host placeholder (public
repo — no hostnames), device identity (chip/driver per receipts
convention), kernel/driver versions, artifact digests, exact commands,
generated IDs, timing separated cold/warm.

## Qualification status semantics

Device PASS (frozen rule above) upgrades the catalog family entry from
unqualified to generation-qualified (recommended still requires the
serve/http split per catalog contract). Until then the artifact stays
unqualified and unrecommended everywhere.
