import hashlib
import json

base = "/var/tmp/ParakeetE2EBaseline7d82"
dirs = {1: "out-rr1"}
dirs.update({i: f"out-r{i}" for i in range(2, 7)})
for i, o in sorted(dirs.items()):
    d = json.load(open(f"{base}/{o}/e2e-report.json"))
    seq = d["layers"]["layer_6_decoder_sequence"]
    enc = d["layers"]["layer_5_encoder"]
    mel = d["layers"]["layer_2_preprocessing"]
    txt = d["layers"]["layer_7_end_to_end_text"]
    O = f"{base}/{o}/"
    tr = hashlib.sha256(open(O + "transcript.txt", "rb").read()).hexdigest()
    eh = hashlib.sha256(open(O + "encoder_hidden.npy", "rb").read()).hexdigest()
    ml = hashlib.sha256(open(O + "mel.npy", "rb").read()).hexdigest()
    print(
        f"r{i}: status={d['status']} "
        f"emissions={seq['actual_emissions']}/{seq['native_emissions']} "
        f"prefix={seq['matching_prefix_length']} tokens={seq['tokens_match']} "
        f"durations={seq['durations_match']} frames={seq['frame_indices_match']} "
        f"bounds={enc['all_bounds_pass']} mel_exact={mel['mel']['bit_exact']} "
        f"transcript={txt['transcript_match']}"
    )
    print(f"   transcript={tr[:16]} encoder_hidden={eh[:16]} mel={ml[:16]}")
