#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/horde/repos/isaaclab-dvi-prev-dvi-paper-lfsless-test
NEWTON=/home/horde/repos/newton-dvi-prev-dvi-paper-lfsless-test
STATE=/home/horde/.openclaw/workspace/memory/checkpoint-regeneration-state.md
SUPLOG="$ROOT/logs/checkpoint_regeneration_supervisor.log"
PY=/home/horde/miniforge3/envs/dvi/bin/python
SOURCE_PATH=$(find "$ROOT/source" -mindepth 1 -maxdepth 1 -type d -printf '%p:')
export PYTHONPATH="${SOURCE_PATH}${NEWTON}"
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
exec >>"$SUPLOG" 2>&1

ts(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log(){ echo "[$(ts)] $*"; }
fail(){ log "BLOCKED: $*"; update_state "[blocked] $*"; exit 1; }
update_state(){
  local status="$1"
  cat >"$STATE" <<EOF
# Sequential DVI/Jacobi checkpoint regeneration

- Supervisor: tmux \`checkpoint-regen-supervisor\`, script \`$ROOT/checkpoint_regeneration_supervisor.sh\`
- Supervisor log: \`$SUPLOG\`
- Updated: $(ts)
- Current: $status
- H1: validated complete at \`$ROOT/logs/rsl_rl/h1_flat/2026-09-11_02-57-55_h1_dvi_jacobi_old_branches_full_20260911/model_999.pt\` (checkpoint iter/TensorBoard step 999, success 1.0, load/playback smoke passed).
- Sequence/settings recovered from \`/home/horde/repos/Newton-DVI-tech-report/results/reinforcement_learning/{parameters.csv,RL_SOURCE_MANIFEST.csv,raw/**/run.log\`.
- Settings: Ant 1000/20/c1/post=true/substeps1; Humanoid 1000/50/c1/post=true/substeps2; Go2 500/20/c2/post=false/substeps2; ANYmal-C 500/40/c2/post=false/substeps1; G1 1500/40/c2/post=false/substeps1. H1 uses 1000/40/c1/post=false/substeps1. All 4096 envs, seed 42, sparse Jacobi contacts, sparse LDL joints, factorization cache enabled.
- No commit/push/reset/checkout/stash/delete operations are performed.
EOF
}

# id|task|experiment|max_iters|expected_last|config assertions
after_h1=(
'ant|Isaac-Ant-Direct-v0|ant_direct|1000|999|20|1|True|1'
'humanoid|Isaac-Humanoid-Direct-v0|humanoid_direct|1000|999|50|1|True|2'
'go2|Isaac-Velocity-Flat-Unitree-Go2-v0|unitree_go2_flat|500|499|20|2|False|2'
'anymal-c|Isaac-Velocity-Flat-Anymal-C-v0|anymal_c_flat|500|499|40|2|False|1'
'g1|Isaac-Velocity-Flat-G1-v0|g1_flat|1500|1499|40|2|False|1'
)

validate_run(){
  local id=$1 task=$2 run=$3 expected=$4 ci=$5 coupling=$6 post=$7 substeps=$8
  [[ -s "$run/model_${expected}.pt" ]] || fail "$id missing model_${expected}.pt in $run"
  RUN="$run" EXPECTED="$expected" CI="$ci" COUPLING="$coupling" POST="$post" SUBSTEPS="$substeps" "$PY" - <<'PY'
import os, torch, yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
r=os.environ['RUN']; expected=int(os.environ['EXPECTED'])
x=torch.load(f'{r}/model_{expected}.pt',map_location='cpu',weights_only=False)
assert x.get('iter') == expected, (x.get('iter'), expected)
e=EventAccumulator(r); e.Reload()
mx=max(v.step for t in e.Tags()['scalars'] for v in e.Scalars(t))
assert mx >= expected, (mx, expected)
with open(f'{r}/params/env.yaml') as f: cfg=yaml.load(f, Loader=yaml.UnsafeLoader)
p=cfg['sim']['physics']; s=p['solver_cfg']
assert p['num_substeps']==int(os.environ['SUBSTEPS'])
assert s['contact_solver_type']=='sparse_jacobi'
assert s['joint_solver_type']=='sparse_ldl'
assert s['contact_max_iterations']==int(os.environ['CI'])
assert s['coupling_iterations']==int(os.environ['COUPLING'])
assert s['cache_factorization'] is True
assert s['post_stabilize_joints'] is (os.environ['POST']=='True')
print('VALID',r,'iter',expected,'tb',mx)
PY
  local plog="$run/playback_smoke.log"
  set +e
  timeout --signal=INT 55s "$PY" scripts/reinforcement_learning/rsl_rl/play.py \
    --task "$task" --checkpoint "$run/model_${expected}.pt" --num_envs 32 --seed 42 \
    presets=newton_dvi env.sim.physics=newton_dvi >"$plog" 2>&1
  local rc=$?
  set -e
  if ! grep -q 'Loading model checkpoint from:' "$plog" || grep -Eq 'Traceback|NaN values|CUDA error' "$plog"; then
    fail "$id checkpoint load/playback smoke failed (rc=$rc, $plog)"
  fi
  [[ $rc -eq 0 || $rc -eq 124 || $rc -eq 130 ]] || fail "$id playback exited unexpectedly rc=$rc"
  log "$id accepted: $run/model_${expected}.pt (playback rc=$rc)"
  printf '%s\n' "$run" > "$ROOT/logs/checkpoint_regeneration_${id}.accepted"
}

update_state 'durable supervisor active; H1 accepted; preparing Ant'
for spec in "${after_h1[@]}"; do
  IFS='|' read -r id task experiment max_iters expected ci coupling post substeps <<<"$spec"
  marker="$ROOT/logs/checkpoint_regeneration_${id}.accepted"
  if [[ -s "$marker" ]]; then log "$id already accepted: $(cat "$marker")"; continue; fi
  update_state "training $id ($task), tmux checkpoint-regen-$id"
  before=$(find "$ROOT/logs/rsl_rl/$experiment" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort || true)
  run_name="paper_jacobi_regen_${id}_$(date -u +%Y%m%dT%H%M%SZ)"
  launcher="$ROOT/logs/checkpoint_regeneration_${id}_launch.sh"
  console="$ROOT/logs/checkpoint_regeneration_${id}_console.log"
  cat >"$launcher" <<EOF
#!/usr/bin/env bash
set -o pipefail
cd '$ROOT'
export PYTHONPATH='$PYTHONPATH'
export CUDA_VISIBLE_DEVICES=0
'$PY' scripts/reinforcement_learning/rsl_rl/train.py --task '$task' --num_envs 4096 --seed 42 --max_iterations '$max_iters' --run_name '$run_name' presets=newton_dvi env.sim.physics=newton_dvi 2>&1 | tee '$console'
rc=\${PIPESTATUS[0]}; echo TRAIN_EXIT=\$rc | tee -a '$console'; exit \$rc
EOF
  chmod +x "$launcher"
  session="checkpoint-regen-$id"
  tmux has-session -t "$session" 2>/dev/null && fail "unexpected existing tmux session $session"
  tmux new-session -d -s "$session" "$launcher"
  log "launched $id in tmux $session"
  while tmux has-session -t "$session" 2>/dev/null; do sleep 60; done
  grep -q 'TRAIN_EXIT=0' "$console" || fail "$id training failed; inspect $console"
  run=$(find "$ROOT/logs/rsl_rl/$experiment" -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" -printf '%p\n' | sort | tail -1)
  [[ -n "$run" ]] || fail "$id completed but run directory was not found"
  validate_run "$id" "$task" "$run" "$expected" "$ci" "$coupling" "$post" "$substeps"
  update_state "$id accepted; advancing sequentially"
done
update_state 'COMPLETE: H1, Ant, MuJoCo Humanoid, Go2, ANYmal-C, and G1 checkpoints all validated'
log 'ALL CHECKPOINTS COMPLETE'
