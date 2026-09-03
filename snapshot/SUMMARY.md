# Community hardware data snapshot

Records: 2 | dataset generated_at: 2026-09-03T18:09:08.000Z | source: https://mlx-omarchy-community-data.joshua-s-warren.workers.dev

Mirrored by `.github/workflows/community-data.yml` from the public
read API. Summaries only; archive blobs stay on the endpoint.

| hash | kind | chip | kernel | mesa | mlx-omarchy | bench |
|---|---|---|---|---|---|---|
| [57a126897870](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/57a126897870d5ac7f9b8349db45618006af529cf7d624daebda4a09526428ae) | quick | apple,t8103 | 7.1.6-1-1-ARCH | Honeykrisp | 0.32.2.dev20260903 | [0](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/57a126897870d5ac7f9b8349db45618006af529cf7d624daebda4a09526428ae/archive) |
| [47bd68fd2bd3](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/47bd68fd2bd38612d72707010e7e2376dd348fca0b0e24e575cca2dc6c5ffe54) | deep | apple,t8103 | 7.1.6-1-1-ARCH | Honeykrisp | 0.32.2.dev20260903 | [6](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/47bd68fd2bd38612d72707010e7e2376dd348fca0b0e24e575cca2dc6c5ffe54/archive) |

Query this snapshot:

```
python3 scripts/query_community_data.py list
python3 scripts/query_community_data.py --json --kind deep list
python3 scripts/query_community_data.py show <sha256-prefix>
python3 scripts/query_community_data.py compare --metric tflops
```

Redacted, allowlisted fields only. Submissions never touch the
contributor graph: this branch is written only by github-actions[bot].
