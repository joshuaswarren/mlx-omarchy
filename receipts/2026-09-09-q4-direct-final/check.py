import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
expected = None
count = 0
for rep in range(1, 6):
    for side in ("baseline", "candidate"):
        directory = root / f"pair-{rep}-{side}"
        captures = [json.loads(line) for line in (directory / "rep1.ids.jsonl").read_text().splitlines()]
        assert len(captures) == 6
        assert all(len(row["ids"]) == row["requested"] for row in captures)
        ids = [row["ids"] for row in captures]
        if expected is None:
            expected = ids
        assert ids == expected, (rep, side)
        record = json.loads((directory / "rep1.json").read_text())
        assert record["clean_check"]["status"] == "clean"
        assert record["power"]["source"] == "AC"
        provenance = record["binary_provenance"]["omarchy"]
        assert provenance["verified"] == "match" and provenance["version_match"]
        legs = [leg for leg in record["legs"] if leg.get("measured")]
        assert len(legs) == 6
        for leg in legs:
            assert leg["exit_code"] == 0 and not leg["contended"]
            for metric in ("decode_tok_s", "prefill_tok_s"):
                value = leg["metrics"][metric]
                assert math.isfinite(value) and value > 0
        count += len(legs)
print(json.dumps({"measured_legs": count, "full_ids_equal": True, "wheel_records_match": True, "power": "AC", "contention": False}))
