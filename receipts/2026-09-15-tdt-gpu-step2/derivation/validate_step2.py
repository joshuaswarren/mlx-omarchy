# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# Unit validation: fused GPU decoder step vs the landed numpy BNNS contract
# and run_joint, bit for bit, on real pinned weights.
import sys
from pathlib import Path

import numpy as np

PKG = Path("/var/tmp/TdtGpuStep2/pkg")
MODEL = Path(
    "/home/joshuawarren/.cache/mlx-omarchy/parakeet-reference/mweinbach1/"
    "parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018"
)


DEBUG_MODE = 0


def main(trials: int = 24) -> int:
    global DEBUG_MODE
    import os
    DEBUG_MODE = int(os.environ.get("TDT_DEBUG_MODE", "0"))
    import mlx.core as mx

    sys.path.insert(0, str(PKG.resolve()))
    sys.path.insert(0, str((PKG / "coreml").resolve()))

    from coreml.vulkan_decoder import load_decoder
    from coreml.vulkan_decoder_step import pack_step_weights, run_step
    from coreml.vulkan_joint import run_joint

    rng = np.random.default_rng(20260915)
    decoder = load_decoder(MODEL / "decoder.mlpackage")
    packed = pack_step_weights(decoder, MODEL / "joint.mlpackage")
    weights = decoder._weights
    embedding = np.asarray(weights.embedding)
    projector = np.asarray(weights.projector, dtype=np.float16)
    projector_bias = np.asarray(weights.projector_bias, dtype=np.float16)

    frames = rng.standard_normal((1, 24, 640)).astype(np.float32)
    with mx.stream(mx.gpu):
        encoder_dev = mx.array(frames)
    mx.eval(encoder_dev)

    hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
    cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(hidden, cell)

    bad = 0
    for trial in range(trials):
        token = int(rng.integers(0, 8193))
        frame = int(rng.integers(0, 24))
        # fresh pseudo-state every few trials, recurrent chain otherwise
        if trial % 6 == 5:
            hidden = mx.array(
                rng.standard_normal((2, 1, 640)).astype(np.float32)
            )
            cell = mx.array(rng.standard_normal((2, 1, 640)).astype(np.float32))
            mx.eval(hidden, cell)
        state_out, token_out, dur_out, logits, dbg = run_step(
            packed, hidden, cell, token, encoder_dev, frame,
            mode=2 if DEBUG_MODE == 2 else None,
        )
        if DEBUG_MODE == 2:
            if trial == 0:
                import coreml.vulkan_decoder as vdd
                x_in = np.asarray(weights.embedding)[token].ravel()
                h0 = np.asarray(hidden).reshape(1280)[:640].astype(np.float16)
                c0 = np.asarray(cell).reshape(1280)[:640].astype(np.float16)
                a = np.concatenate([x_in, h0]).astype(np.float16)
                w64, bias = decoder._layers[0]
                want_preact = vdd._add16(vdd._blocked_gemv(a, w64), np.asarray(bias, np.float16).ravel())
                got = np.asarray(dbg[:320])
                import coreml.vulkan_decoder as vdd
                x_in = np.asarray(weights.embedding)[token].ravel()
                h0 = np.asarray(hidden).reshape(1280)[:640].astype(np.float16)
                c0 = np.asarray(cell).reshape(1280)[:640].astype(np.float16)
                a = np.concatenate([x_in, h0]).astype(np.float16)
                w64, bias0 = decoder._layers[0]
                from coreml.vulkan_decoder import _fused_luts
                sig, tanh_lut = _fused_luts()
                preact = vdd._add16(vdd._blocked_gemv(a, w64), np.asarray(bias0, np.float16).ravel())
                si = sig[preact[:640].view(np.uint16)]
                sf = sig[preact[640:1280].view(np.uint16)]
                so = sig[preact[1280:1920].view(np.uint16)]
                tg = tanh_lut[preact[1920:2560].view(np.uint16)]
                fprod = (sf.astype(np.float64) * c0.astype(np.float64)).astype(np.float16)
                c1 = (fprod.astype(np.float64) + si.astype(np.float64) * tg.astype(np.float64)).astype(np.float16)
                tc = tanh_lut[c1.view(np.uint16)]
                h1 = (so.astype(np.float64) * tc.astype(np.float64)).astype(np.float16)
                segs = {"pr_i": preact[:640], "pr_f": preact[640:1280], "pr_o": preact[1280:1920], "pr_g": preact[1920:2560],
                        "si": si, "sf": sf, "so": so, "tg": tg, "c1": c1, "h1": h1}
                for idx, (nm, want_seg) in enumerate(segs.items()):
                    got_seg = got[idx * 32:(idx + 1) * 32].astype(np.float16)
                    n_bad = int((got_seg != want_seg[:32]).sum())
                    print(f"L0 {nm:5s} {'OK' if n_bad == 0 else 'BAD'} got {got_seg[:2]} want {want_seg[:2]}")
                # layer 1 expectation
                nh0 = h1
                h1s = np.asarray(hidden).reshape(1280)[640:1280].astype(np.float16)
                a1 = np.concatenate([nh0, h1s]).astype(np.float16)
                w1, b1v = decoder._layers[1]
                pre1 = vdd._add16(vdd._blocked_gemv(a1, w1), np.asarray(b1v, np.float16).ravel())
                si1 = sig[pre1[:640].view(np.uint16)]
                sf1 = sig[pre1[640:1280].view(np.uint16)]
                so1 = sig[pre1[1280:1920].view(np.uint16)]
                tg1 = tanh_lut[pre1[1920:2560].view(np.uint16)]
                c0v = np.asarray(cell).reshape(1280)[640:1280].astype(np.float16)
                fp1 = (sf1.astype(np.float64) * c0v.astype(np.float64)).astype(np.float16)
                c11 = (fp1.astype(np.float64) + si1.astype(np.float64) * tg1.astype(np.float64)).astype(np.float16)
                tc1 = tanh_lut[c11.view(np.uint16)]
                h11 = (so1.astype(np.float64) * tc1.astype(np.float64)).astype(np.float16)
                segs1 = {"pr_i": pre1[:640], "pr_f": pre1[640:1280], "pr_o": pre1[1280:1920], "pr_g": pre1[1920:2560],
                         "si": si1, "sf": sf1, "so": so1, "tg": tg1, "c1": c11, "h1": h11}
                sd_got = np.asarray(dbg[0:640]).astype(np.float16)
                acc_got = np.asarray(dbg[640:1280])
                pj_got = np.asarray(dbg[1280:1920]).astype(np.float16)
                acc_p = np.zeros(640, dtype=np.float32)
                for k in range(640):
                    acc_p += h11.astype(np.float32) * projector.T[k].astype(np.float32)
                want_pj16 = (acc_p.astype(np.float16) + projector_bias)
                print("sh_dec bad:", int((sd_got != h11).sum()), "acc bad:", int((acc_got.astype(np.float16) != acc_p.astype(np.float16)).sum()),
                      "pj bad:", int((pj_got != want_pj16).sum()), "/640")
                bad = np.flatnonzero(acc_got.astype(np.float16) != acc_p.astype(np.float16))
                if len(bad):
                    i = int(bad[0])
                    print(f"  acc lane {i}: got {acc_got[i]!r} want {acc_p[i]!r}")
                print("mode2 state_out[640:644]", np.asarray(state_out[640:644]), "want h1_l0", h1[:4])
                print("mode2 state_out[1280:1284]", np.asarray(state_out[1280:1284]), "want h1_l1", h11[:4])
                got1 = np.asarray(dbg[320:640])
                for idx, (nm, want_seg) in enumerate(segs1.items()):
                    got_seg = got1[idx * 32:(idx + 1) * 32].astype(np.float16)
                    n_bad = int((got_seg != want_seg[:32]).sum())
                    print(f"L1 {nm:5s} {'OK' if n_bad == 0 else 'BAD'} got {got_seg[:2]} want {want_seg[:2]}")
                return 0
        got_dec = np.asarray(state_out[0:640])
        got_nh = np.asarray(state_out[640:1920])
        got_nc = np.asarray(state_out[1920:3200])

        # numpy contract expectation
        import coreml.vulkan_decoder as vd

        x_in = embedding[token].ravel()
        states = []
        x = x_in
        for (w64, bias), h_state, c_state in (
            (decoder._layers[0], np.asarray(hidden[0]).astype(np.float16)[0], np.asarray(cell[0]).astype(np.float16)[0]),
            (decoder._layers[1], np.asarray(hidden[1]).astype(np.float16)[0], np.asarray(cell[1]).astype(np.float16)[0]),
        ):
            nh, nc = vd.fused_lstm_layer(x, h_state, c_state, w64, bias)
            states.append((nh, nc))
            x = nh
        nh1 = states[1][0]
        acc = np.zeros(640, dtype=np.float32)
        for k in range(640):
            acc += nh1.astype(np.float32) * projector.T[k].astype(np.float32)
        acc2 = np.zeros(640, dtype=np.float32)
        hf2 = nh1.astype(np.float32)
        Pt32 = projector.T.astype(np.float32)
        for k in range(640):
            acc2 += hf2[k] * Pt32[k]
        acc3 = np.einsum("k,nk->n", hf2, Pt32)
        if trial == 0:
            print("BISECT acc[:2]", acc[:2], "acc2[:2]", acc2[:2], "acc3[:2]", acc3[:2])
        want_dec16 = (acc2.astype(np.float16) + projector_bias)
        want_dec = want_dec16.astype(np.float32)

        def argmax_first(v):
            return int(np.argmax(v))

        # joint via the landed GPU path
        joint_out = run_joint(
            encoder_dev[:, frame, :],
            mx.array(want_dec).reshape(1, 640),
            package_path=MODEL / "joint.mlpackage",
        )
        tok_logits = np.asarray(joint_out.token_logits).reshape(-1)
        dur_logits = np.asarray(joint_out.duration_logits).reshape(-1)
        want_tok = argmax_first(tok_logits)
        want_dur = argmax_first(dur_logits)

        if trial == 0 and DEBUG_MODE == 0:
            print("VAL nh1[:3]", nh1[:3], "want_dec16[:3]", want_dec16[:3], "got_dec[:3]", got_dec[:3].astype(np.float16))
            print("VAL projector[:2,:2]", projector[:2, :2], "pb[:3]", projector_bias[:3], "acc[:3]", acc[:3])
            import hashlib
            print("VAL nh1 sha", hashlib.sha256(nh1.tobytes()).hexdigest()[:16], "P sha", hashlib.sha256(projector.tobytes()).hexdigest()[:16])
        checks = {
            "next_hidden": np.array_equal(got_nh, np.concatenate([states[0][0], states[1][0]]).astype(np.float32)),
            "next_cell": np.array_equal(got_nc, np.concatenate([states[0][1], states[1][1]]).astype(np.float32)),
            "decoder_hidden": np.array_equal(got_dec, want_dec),
            "logits_token": np.array_equal(np.asarray(logits[:8193]), tok_logits),
            "logits_dur": np.array_equal(np.asarray(logits[8193:8198]), dur_logits),
            "token": token_out == want_tok,
            "duration": dur_out == want_dur,
        }
        # chain the recurrence like the real decode loop
        hidden = state_out[640:1920].reshape(2, 1, 640)
        cell = state_out[1920:3200].reshape(2, 1, 640)
        if not all(checks.values()):
            bad += 1
            fails = [k for k, v in checks.items() if not v]
            print(f"trial {trial} token {token} frame {frame}: FAIL {fails}")
            if trial == 0:
                want_nh = np.concatenate([states[0][0], states[1][0]]).astype(np.float32)
                want_nc = np.concatenate([states[0][1], states[1][1]]).astype(np.float32)
                print("  got_nh[:4]", got_nh[:4], "want", want_nh[:4])
                print("  got_nh[640:644]", got_nh[640:644], "want", want_nh[640:644])
                print("  got_nc[:4]", got_nc[:4], "want", want_nc[:4])
                print("  got_dec[:4]", got_dec[:4], "want", want_dec[:4])
                print("  finite:", np.isfinite(got_nh).all(), np.isfinite(got_nc).all())
            if "decoder_hidden" not in checks or not checks["decoder_hidden"]:
                d = np.abs(got_dec.astype(np.float64) - want_dec.astype(np.float64))
                print(f"  dec max_abs {d.max()}")
    print(f"VALIDATION {'PASS' if bad == 0 else 'FAIL'} ({bad}/{trials} trials failed)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
