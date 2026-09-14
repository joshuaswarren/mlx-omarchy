| arm | run | QmmPrefillCoopmatF16 ms | n | prefill-window GPU busy ms | host prefill ms | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 705.205 | 163 | 931.183 | 1292.2 | 814.9 |
| base | 2 | 708.198 | 163 | 943.208 | 1530.8 | 687.9 |
| base | 3 | 708.185 | 163 | 941.857 | 1926.7 | 546.5 |
| base | 4 | 710.725 | 163 | 947.900 | 1463.1 | 719.7 |
| patched | 1 | 701.372 | 163 | 929.774 | 1332.3 | 790.4 |
| patched | 2 | 709.185 | 163 | 948.296 | 1578.6 | 667.0 |
| patched | 3 | 708.487 | 163 | 946.696 | 1908.9 | 551.6 |
| patched | 4 | 708.044 | 163 | 940.250 | 1499.4 | 702.3 |

QmmPrefillCoopmatF16 GPU time (ms):
   base    median   708.191  range   705.205 ..   710.725
   patched median   708.265  range   701.372 ..   709.185
   patched vs base: -0.01 %   (within-arm base spread 0.8 %)
   -> delta is smaller than the within-arm spread: no resolvable difference

prefill-window GPU busy (ms):
   base    median   942.533  range   931.183 ..   947.900
   patched median   943.473  range   929.774 ..   948.296
   patched vs base: -0.10 %   (within-arm base spread 1.8 %)
   -> delta is smaller than the within-arm spread: no resolvable difference

end-to-end prefill (host wall clock) (tok/s):
   base    median   703.790  range   546.539 ..   814.890
   patched median   684.665  range   551.633 ..   790.366
   patched vs base: -2.72 %   (within-arm base spread 38.1 %)
   -> delta is smaller than the within-arm spread: no resolvable difference

runs per arm: 4 base / 4 patched
QmmPrefillCoopmatF16 dispatch count identical across every run: True [163]
prefill dispatch count identical across every run: True [1063]
