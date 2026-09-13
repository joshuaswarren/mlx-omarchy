# Frontend pre-expansion of constexpr_lut_to_dense (2026-09-13)

Stream 1 of the encoder coverage gaps (§19 execution): the mlx-omarchy
Core ML frontend now expands every palettized weight into a dense
constant before compiler input. The compiler registry is untouched — no
op was added to mil-hwx-compiler, per the ownership rule.

## Semantics provenance (all read from Apple sources this session)

- Op contract: coremltools `iOS18/compression.py`
  `constexpr_lut_to_dense` (lut rank = indices rank + 2; contiguous
  per-axis palette blocks — `np.repeat` tile semantics; scalar palettes
  only when `vector_axis` is absent; output dtype/shape = lut dtype /
  indices shape) and `optimize/_utils.py::lut_to_dense`.
- Blob storage: `MILBlob/Blob/StorageFormat.hpp` — 64-byte-aligned
  `storage_header` (count, version=2) then `blob_metadata`/data pairs;
  `Value.blobFileValue.offset` points at the metadata struct (sentinel
  0xDEADBEEF, dtype u32, sizeInBytes, data offset, padding bits).
- Blob dtype codes: `BlobDataType.hpp` (Float16=1, Float32=2, UInt4=11).
- uint4 packing: `SubByteTypes.cpp::PackSubByteVecImpl` — element `i`
  at bit offset `nbits * (i % elements_per_byte)` within byte
  `i // elements_per_byte`: element 2k = low nibble, 2k+1 = high
  nibble. Matches `np.unpackbits(bitorder="little")`.
- Empirical confirmation on the pinned encoder: indices blob forward
  spans are 2,097,216 B for 4,194,304-element uint4 tensors (nibble
  packing + 64 B alignment), lut spans 2,048 B + 64 for 1,024-entry
  fp16 palettes.

## Implementation

`overlay/tools/coreml/depalettize.py`:

- `read_blob_metadata` — parse/validate one blob metadata entry.
- `unpack_packed_uints` — LSB-first sub-byte unpack (uint1/2/3/4/6/8).
- `expand_lut_constants(spec, weights_dir, expansion_path)` — walk all
  functions/block specializations/nested blocks; for each
  `constexpr_lut_to_dense`: resolve the blob-backed `indices`/`lut`
  bindings, validate the coremltools shape contract (named
  `DepalettizeError` on every violation, including vector
  palettization), gather raw fp16/fp32 payload bytes (uint16/uint32
  view — no float arithmetic, NaN/±0/inf payloads preserved), append
  the dense tensor to `expansion_path` as a properly aligned blob
  entry, and rewrite the op in place into a `const` whose `val`
  attribute points at the new blob. Output names are unchanged, so all
  consumers keep resolving.

## Verification

Host tests (`overlay/tests/omarchy/coreml/test_depalettize.py`, 4/4 OK):

1. uint4 nibble order matches the C++ pack loop (0x21,0xF0,0x84 →
   1,2,0,15,4,8).
2. Synthetic small palettized package (uint4 indices [8,12], fp16 lut
   [2,1,16,1], palette entries covering +0, −0, +inf, −inf, NaN
   payloads): the expanded const equals an independent pure-Python
   gather byte for byte; the transformed package's op histogram is
   `{const: 1}` — no `constexpr_lut_to_dense`.
3. Named errors: vector palettization, non-divisible indices dim.
4. Named error: wrong blob sentinel.

Real pinned encoder (`encoder-expansion.log`, package copy under
`.work/expanded-encoder/`, cache untouched):

- before: 3351 ops / 29 types, `constexpr_lut_to_dense` 194, `const` 1783
- expanded 194 ops in 13.6 s, 1,016,332,288 dense bytes appended
  (weight.bin 444,016,768 → 1,460,361,472 bytes; copy verified against
  the lock-pinned weight SHA-256
  `23867a834223ee6484d0b7b9b703530e8606546efa8e710535163d51ed9aac04`
  before any mutation)
- after: 3351 ops / 28 types, `constexpr_lut_to_dense` 0, `const` 1977
- three sampled expanded tensors (incl. the [4096,1024] indices with
  [256,1,16,1] lut variant) byte-equal an independent cross-check that
  runs the literal coremltools algorithm (`np.repeat` + `np.take`)
- dense fp16 sanity: all finite, max |w| = 4.1641

## Remaining encoder gaps (mil-hwx-compiler backlog — not implemented here)

- Stream 2, layout/data movement (219 ops): `transpose` (146,
  IR-known, no H13 lowering branch), `slice_by_index` (48), `pad`
  (24), `tile` (1).
- Stream 3, boolean/mask path (72 ops): `select` (48), `cast` (11),
  `less` (4), `floor` (3), `floor_div` (3), `logical_and` (1),
  `logical_not` (1), `reduce_min` (1) — needs an fp16 mask
  representation decision in the compiler repo.

These stay owned by mil-hwx-compiler per plan §19.
