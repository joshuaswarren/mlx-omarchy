# Native macOS Metal baselines vs Linux canonical 12-matrix

Linux legs: fork/stock on jwm1-linux (base M1, T8103, 8-core GPU), commit b6d662a8.

**No macOS host measured is a base M1.** All native rows are cross-chip and are not a same-chip denominator for the Linux host. Per-chip numbers, unnormalized.

| leg (prompt/gen) | host (chip) | decode med tok/s | prefill med tok/s | d vs fork | d vs stock | p vs fork | p vs stock | digest =ref |
|---|---|---|---|---|---|---|---|---|
| q4_short (30/32) | jwm1-linux fork (base M1) | 112.195 | 331.502 | 1.00 | — | 1.00 | — | — |
| q4_short (30/32) | jwm1-linux stock (base M1) | 102.525 | 170.942 | — | 1.00 | — | 1.00 | — |
| q4_short (30/32) | 16M1MBP (Apple M1 Max) | 286.961 (282.954–300.485) | 1517.553 (1427.436–1791.486) | 2.558x | 2.799x | 4.578x | 8.878x | yes |
| q4_short (30/32) | MacStudio (Apple M1 Ultra) | 299.192 (285.779–304.541) | 1410.553 (1129.874–1507.392) | 2.667x | 2.918x | 4.255x | 8.252x | yes |

| q4_long (262/128) | jwm1-linux fork (base M1) | 108.945 | 970.37 | 1.00 | — | 1.00 | — | — |
| q4_long (262/128) | jwm1-linux stock (base M1) | 81.875 | 310.06 | — | 1.00 | — | 1.00 | — |
| q4_long (262/128) | 16M1MBP (Apple M1 Max) | 296.253 (294.511–300.425) | 5937.907 (5735.819–5994.429) | 2.719x | 3.618x | 6.119x | 19.151x | yes |
| q4_long (262/128) | MacStudio (Apple M1 Ultra) | 287.093 (281.958–290.614) | 6754.282 (6092.722–7460.376) | 2.635x | 3.506x | 6.961x | 21.784x | yes |

| q4_1K ctx (1053/32) | jwm1-linux fork (base M1) | 96.125 | 1112.52 | 1.00 | — | 1.00 | — | — |
| q4_1K ctx (1053/32) | jwm1-linux stock (base M1) | 52.91 | 361.671 | — | 1.00 | — | 1.00 | — |
| q4_1K ctx (1053/32) | 16M1MBP (Apple M1 Max) | 283.794 (283.255–285.578) | 8048.422 (7943.585–8380.355) | 2.952x | 5.364x | 7.234x | 22.253x | yes |
| q4_1K ctx (1053/32) | MacStudio (Apple M1 Ultra) | 282.982 (243.017–289.601) | 11040.136 (3796.071–11484.302) | 2.944x | 5.348x | 9.924x | 30.525x | yes |

| bf16_short (30/32) | jwm1-linux fork (base M1) | 11.7 | 85.96 | 1.00 | — | 1.00 | — | — |
| bf16_short (30/32) | jwm1-linux stock (base M1) | 11.735 | 83.799 | — | 1.00 | — | 1.00 | — |
| bf16_short (30/32) | 16M1MBP (Apple M1 Max) | 217.882 (215.946–220.63) | 1192.738 (1133.66–1444.104) | 18.622x | 18.567x | 13.876x | 14.233x | yes |
| bf16_short (30/32) | MacStudio (Apple M1 Ultra) | 257.078 (250.772–259.213) | 1111.172 (1004.333–1425.779) | 21.972x | 21.907x | 12.927x | 13.26x | yes |

| bf16_long (262/128) | jwm1-linux fork (base M1) | 11.255 | 317.769 | 1.00 | — | 1.00 | — | — |
| bf16_long (262/128) | jwm1-linux stock (base M1) | 11.29 | 210.78 | — | 1.00 | — | 1.00 | — |
| bf16_long (262/128) | 16M1MBP (Apple M1 Max) | 214.715 (210.826–216.763) | 5281.122 (4993.385–5333.899) | 19.077x | 19.018x | 16.619x | 25.055x | yes |
| bf16_long (262/128) | MacStudio (Apple M1 Ultra) | 252.921 (251.759–254.286) | 6265.091 (6189.738–6399.231) | 22.472x | 22.402x | 19.716x | 29.723x | yes |

| bf16_1K ctx (1053/32) | jwm1-linux fork (base M1) | 10.03 | 364.802 | 1.00 | — | 1.00 | — | — |
| bf16_1K ctx (1053/32) | jwm1-linux stock (base M1) | 10.125 | 220.132 | — | 1.00 | — | 1.00 | — |
| bf16_1K ctx (1053/32) | 16M1MBP (Apple M1 Max) | 207.954 (201.086–210.728) | 7751.351 (7274.779–7903.343) | 20.733x | 20.539x | 21.248x | 35.212x | yes |
| bf16_1K ctx (1053/32) | MacStudio (Apple M1 Ultra) | 243.385 (241.965–244.779) | 9932.535 (9777.261–10321.236) | 24.266x | 24.038x | 27.227x | 45.121x | yes |

