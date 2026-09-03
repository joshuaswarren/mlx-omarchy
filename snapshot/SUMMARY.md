# Community hardware data snapshot

Records: 2 | dataset generated_at: 2026-09-03T17:51:51.000Z | source: https://mlx-omarchy-community-data.joshua-s-warren.workers.dev

Mirrored by `.github/workflows/community-data.yml` from the public
read API. Summaries only; archive blobs stay on the endpoint.

| hash | kind | chip | kernel | mesa | mlx-omarchy | bench |
|---|---|---|---|---|---|---|
| [d9fc365c88fa](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/d9fc365c88fa6f59bccf6f05c27f54170599f06f72cb2caf81498b8b390141bc) | quick | apple,t8103 | 7.1.6-1-1-ARCH | Honeykrisp | 0.32.2.dev202609030512 | [0](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/d9fc365c88fa6f59bccf6f05c27f54170599f06f72cb2caf81498b8b390141bc/archive) |
| [f4728d2b5e0d](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/f4728d2b5e0d4b08ae933edfc18a5005c59839064986272efaf791c58aa5d98e) | deep | apple,t8103 | 7.1.6-1-1-ARCH | Honeykrisp | 0.32.2.dev202609030512 | [6](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/f4728d2b5e0d4b08ae933edfc18a5005c59839064986272efaf791c58aa5d98e/archive) |

Query this snapshot:

```
python3 scripts/query_community_data.py list
python3 scripts/query_community_data.py --json --kind deep list
python3 scripts/query_community_data.py show <sha256-prefix>
python3 scripts/query_community_data.py compare --metric tflops
```

Redacted, allowlisted fields only. Submissions never touch the
contributor graph: this branch is written only by github-actions[bot].
