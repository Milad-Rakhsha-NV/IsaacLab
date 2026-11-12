# G1 rigid, RFT and MPM study

G1 training, acceptance checks, eight paired-speed videos and material plots completed

Both stages: 4,096 environments, seed 42, 1,500 PPO iterations. Forward commands: 0.1–1.0 m/s.
Rigid trains from scratch; RFT fine-tunes the accepted rigid policy for 1,500 additional iterations.
No command curriculum. RFT materials change at reset; layers start at 0–10 mm and widen to 0–118 mm.
MPM is inference only; numerical convergence and material equivalence remain unvalidated.

[Report](report/index.html)

| Trial | Mean tracking error | First failure threshold |
|---|---:|---|
| Rigid-trained policy on a rigid floor (1.00 m/s) | 0.058 m/s | None |
| Rigid-trained policy on a rigid floor (0.50 m/s) | 0.029 m/s | None |
| RFT-trained policy on RFT sand (1.00 m/s) | 0.154 m/s | None |
| RFT-trained policy on RFT sand (0.50 m/s) | 0.097 m/s | None |
| RFT-trained policy on MPM sand (1.00 m/s) | 0.940 m/s | 1.14 s |
| RFT-trained policy on MPM sand (0.50 m/s) | 0.493 m/s | 1.74 s |
| Rigid-trained policy on MPM sand (1.00 m/s) | 0.943 m/s | 0.90 s |
| Rigid-trained policy on MPM sand (0.50 m/s) | 0.487 m/s | 1.18 s |
