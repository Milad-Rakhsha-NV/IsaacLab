#!/bin/bash
cd /home/horde/repos/IsaacLab
eval "$(~/miniforge3/bin/conda shell.bash hook)" && conda activate isaaclab-chrono

echo "=== ANYmal-C Chrono Standing Parameter Sweep (500 steps = 2.5s sim) ==="
echo ""
printf "%15s %8s %8s %8s | %8s %8s %8s\n" "recovery_spd" "gap" "margin" "substeps" "final_h" "standing" "min_h"
echo "------------------------------------------------------------------------"

run_test() {
    local rs=$1 gap=$2 margin=$3 sub=$4
    result=$(python test_anymal_standing.py --rs $rs --gap $gap --margin $margin --steps 500 --substeps $sub 2>&1 | grep "Final height")
    final_h=$(echo "$result" | grep -oP 'Final height: \K[0-9.-]+')
    standing=$(echo "$result" | grep -oP 'Standing: \K\w+')
    min_result=$(python test_anymal_standing.py --rs $rs --gap $gap --margin $margin --steps 500 --substeps $sub 2>&1 | grep "Height range" | grep -oP '\[\K[0-9.-]+')
    printf "%15s %8s %8s %8s | %8s %8s\n" "$rs" "$gap" "$margin" "$sub" "$final_h" "$standing"
}

# Actually, let's just run each and capture both lines
run_one() {
    local rs=$1 gap=$2 margin=$3 sub=$4
    output=$(python test_anymal_standing.py --rs $rs --gap $gap --margin $margin --steps 500 --substeps $sub 2>&1)
    final_line=$(echo "$output" | grep "Final height")
    range_line=$(echo "$output" | grep "Height range")
    printf "%12s %8s %8s %4s | %s | %s\n" "$rs" "$gap" "$margin" "$sub" "$final_line" "$range_line"
}

echo ""
echo "--- Recovery speed sweep (gap=0.01, margin=0.001, sub=4) ---"
for rs in 0.1 1.0 10.0 100.0 1000.0 10000.0; do
    run_one $rs 0.01 0.001 4
done

echo ""
echo "--- Gap sweep (rs=1.0, margin=0.001, sub=4) ---"
for gap in 0.0 0.001 0.005 0.01 0.02 0.05; do
    run_one 1.0 $gap 0.001 4
done

echo ""
echo "--- Margin sweep (rs=1.0, gap=0.01, sub=4) ---"
for margin in 0.0 0.001 0.005 0.01 0.02; do
    run_one 1.0 0.01 $margin 4
done

echo ""
echo "--- Gap sweep with high recovery (rs=1000, margin=0.001, sub=4) ---"
for gap in 0.0 0.001 0.005 0.01 0.02 0.05; do
    run_one 1000.0 $gap 0.001 4
done

echo ""
echo "--- Substeps sweep (rs=1.0, gap=0.01, margin=0.001) ---"
for sub in 1 2 4 8; do
    run_one 1.0 0.01 0.001 $sub
done

echo ""
echo "=== SWEEP COMPLETE ==="
