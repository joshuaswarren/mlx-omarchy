# Portable mel comparison: exact parity remains open

The diagnostic compares the pinned LibriSpeech waveform against the accepted
macOS mel capture. It verifies all five input-file SHA-256 values against
`parakeet-reference.lock` before computing a comparison, rejects non-finite
arrays or mismatched shapes/dtypes, and compares float32 bit patterns.

Source algorithm: `mweinbach/parakeet-coreml-swift` at
`75aec2a1c991319657ff4dec5f602c12da6c5012`, Apache-2.0. The diagnostic implementation
is on `parakeet-mel-exact`; it does not become a shipping frontend on main.

## Measured result

```text
shape: expected (3001, 128), computed (3001, 128)
exact golden equality: False
bit-identical values: 215401/384128 (56.08%)
max |diff|: 4.769862e-05
mean |diff|: 1.532242e-07
first difference: frame 0, bin 0
  golden:   -0.1173354983329773
  computed: -0.1173354908823967
```

Shapes and masks match. Encoder input values do not. The diagnostic exits 1;
this is evidence of the remaining mismatch, not a passing parity check.

The tested precision variants do not isolate the cause to the FFT or prove
anything about the reference compiler's FMA contraction. Stage-level reference
comparisons remain necessary. No claim of impossible portable reproduction is
supported by this experiment.

Run `python3 overlay/tools/coreml/mel_reference.py <pinned-capture-directory>`.
The accepted capture is `20260912T154759Z-librispeech/ane` under the reference
cache. No tolerances, golden outputs, or compiler/runtime contracts changed.
The experimental unit tests were removed, including one that required the
frontend to remain wrong. The executable diagnostic is the experiment check.
