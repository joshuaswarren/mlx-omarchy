# Community hardware data snapshot

Records: 1 | dataset generated_at: 2026-09-03T16:48:48.000Z | source: https://mlx-omarchy-community-data.joshua-s-warren.workers.dev

Mirrored by `.github/workflows/community-data.yml` from the public
read API. Summaries only; archive blobs stay on the endpoint.

| hash | kind | chip | kernel | mesa | mlx-omarchy | bench |
|---|---|---|---|---|---|---|
| [75ab079790a4](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/75ab079790a4a357e1c9b6a1775f606fcfbdaababa94b5ccf1fb92dea3d7663c) | quick | apple,t8103 | 7.1.6-1-1-ARCH | Honeykrisp | 0.32.2.dev202609030512 | [0](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/75ab079790a4a357e1c9b6a1775f606fcfbdaababa94b5ccf1fb92dea3d7663c/archive) |

Query this snapshot:

```
python3 scripts/query_community_data.py list
python3 scripts/query_community_data.py --json --kind deep list
python3 scripts/query_community_data.py show <sha256-prefix>
python3 scripts/query_community_data.py compare --metric tflops
```

Redacted, allowlisted fields only. Submissions never touch the
contributor graph: this branch is written only by github-actions[bot].
