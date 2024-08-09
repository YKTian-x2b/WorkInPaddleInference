we are tunning for problem: [8192, 4096, 13696]
fp16_matmul_add: tactic 0 has max diff 0.00195312 compared with baseline,cost_time: 109.896ms.
fp16_matmul_add: tactic 3 has max diff 0.00195312 compared with baseline,cost_time: 88.919ms.
fp16_matmul_add: tactic 6 has max diff 0.00195312 compared with baseline,cost_time: 84.4483ms.
fp16_matmul_add: tactic 12 has max diff 0.00195312 compared with baseline,cost_time: 82.9819ms.
fp16_matmul_add: tactic 15 has max diff 0.00195312 compared with baseline,cost_time: 81.8995ms.


we are tunning for problem: [8, 4096, 13696]
fp16_matmul_add: tactic 0 has max diff 0.000488281 compared with baseline,cost_time: 3.43245ms.
fp16_matmul_add: tactic 2 has max diff 0.00195312 compared with baseline,cost_time: 2.74432ms.
fp16_matmul_add: tactic 5 has max diff 0.00195312 compared with baseline,cost_time: 2.72282ms.
fp16_matmul_add: tactic 10 has max diff 0.00195312 compared with baseline,cost_time: 1.76333ms.
fp16_matmul_add: tactic 16 has max diff 0.00195312 compared with baseline,cost_time: 1.76128ms.
fp16_matmul_add: tactic 37 has max diff 0.00195312 compared with baseline,cost_time: 1.4336ms.
fp16_matmul_add: tactic 81 has max diff 0.000488281 compared with baseline,cost_time: 1.33939ms.


ncu --nvtx --nvtx-exclude "InitMatrix" -f -o profile_for_8_4096_13693_k2 --set detailed ./splitK