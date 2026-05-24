#!/bin/bash
cd /home/horde/repos/IsaacLab
eval "$(~/miniforge3/bin/conda shell.bash hook)" && conda activate isaaclab-chrono

echo "=== Go2 Standing Comparison: Chrono vs MuJoCo (1000 steps = 5s sim) ==="

echo ""
echo "--- MuJoCo baseline ---"
python test_go2_standing.py --solver mujoco --steps 1000 2>&1 | grep -E "Solver:|Step|^[[:space:]]+[0-9]|Final|Height"

echo ""
echo "--- Chrono default (rs=1.0, gap=0.01, margin=0.001, sub=1) ---"
python test_go2_standing.py --solver chrono --rs 1.0 --gap 0.01 --margin 0.001 --substeps 1 --steps 1000 2>&1 | grep -E "Solver:|Step|^[[:space:]]+[0-9]|Final|Height"

echo ""
echo "--- Chrono rs=100 ---"
python test_go2_standing.py --solver chrono --rs 100.0 --gap 0.01 --margin 0.001 --substeps 1 --steps 1000 2>&1 | grep -E "Solver:|Step|^[[:space:]]+[0-9]|Final|Height"

echo ""
echo "--- Chrono rs=1000 ---"
python test_go2_standing.py --solver chrono --rs 1000.0 --gap 0.01 --margin 0.001 --substeps 1 --steps 1000 2>&1 | grep -E "Solver:|Step|^[[:space:]]+[0-9]|Final|Height"

echo ""
echo "--- Chrono gap=0.0 ---"
python test_go2_standing.py --solver chrono --rs 1.0 --gap 0.0 --margin 0.001 --substeps 1 --steps 1000 2>&1 | grep -E "Solver:|Step|^[[:space:]]+[0-9]|Final|Height"

echo ""
echo "--- Chrono gap=0.02 ---"
python test_go2_standing.py --solver chrono --rs 1.0 --gap 0.02 --margin 0.001 --substeps 1 --steps 1000 2>&1 | grep -E "Solver:|Step|^[[:space:]]+[0-9]|Final|Height"

echo ""
echo "--- Chrono margin=0.01 ---"
python test_go2_standing.py --solver chrono --rs 1.0 --gap 0.01 --margin 0.01 --substeps 1 --steps 1000 2>&1 | grep -E "Solver:|Step|^[[:space:]]+[0-9]|Final|Height"

echo ""
echo "--- Chrono substeps=2 ---"
python test_go2_standing.py --solver chrono --rs 1.0 --gap 0.01 --margin 0.001 --substeps 2 --steps 1000 2>&1 | grep -E "Solver:|Step|^[[:space:]]+[0-9]|Final|Height"

echo ""
echo "--- Chrono substeps=4 ---"
python test_go2_standing.py --solver chrono --rs 1.0 --gap 0.01 --margin 0.001 --substeps 4 --steps 1000 2>&1 | grep -E "Solver:|Step|^[[:space:]]+[0-9]|Final|Height"

echo ""
echo "--- Chrono rs=1000, margin=0.01, gap=0.02 ---"
python test_go2_standing.py --solver chrono --rs 1000.0 --gap 0.02 --margin 0.01 --substeps 1 --steps 1000 2>&1 | grep -E "Solver:|Step|^[[:space:]]+[0-9]|Final|Height"

echo ""
echo "=== DONE ==="
