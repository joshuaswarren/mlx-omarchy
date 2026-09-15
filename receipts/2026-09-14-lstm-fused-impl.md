# Fused LSTM impl live decode (2026-09-14)

Host proof 640/640 on reduction-free layer-0 token-8192 (zero embedding,
gates = fp16 bias). Tables sha256
`1ae1e59e186ef1932a54f749c8abf6a4688ffa1dfeab993174502bc7d13d01ae`.

Live native-encoder decode on jwm1-linux with those tables in
`vulkan_decoder.py` **does not land**.

Command: `flock -w 900 /tmp/m1-gpu.lock` +
`/tmp/vulkan-tdt-142/diagnose_decision_142.py` after scp of
`vulkan_decoder.py` (`1a9e9b58…`) and `unary_tables.npz` (`1ae1e59e…`).
Device `Apple M1 (G13G B1)` Honeykrisp. mlx
`0.32.2.dev202609121216+8f6de34c`. Wall 3.97 s. Lock not stolen.

`free_decode.first_divergence`: emission_index **23** (token 939 both,
duration 2 vs native 3, frame 56). Worse than correctly-rounded, which
first-missed at emission 101.

`decoder_on_native_inputs` next_cell not bit-exact (trace 0 max_abs
0.917). Joint on native decoder state still 16/16 token matches.

Decoder **not** landed. Overlay `vulkan_decoder.py` on origin/main stays
the correctly-rounded contract (`c9f78a9c`).

Named hole: fused-op tables fitted on the reduction-free 640-lane slice
do not transfer to the recurrent decode. Next fit must use the 16
native-input decoder traces, not only token-8192.

resolved_model: this session (`xai-oauth/grok-4.6`).
