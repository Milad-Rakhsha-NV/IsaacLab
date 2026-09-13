#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/horde/repos/isaaclab-dvi-prev-dvi-paper-lfsless-test
NEWTON=/home/horde/repos/newton-dvi-prev-dvi-paper-lfsless-test
PY=/home/horde/miniforge3/envs/dvi/bin/python
OUT="$ROOT/logs/checkpoint_regeneration/videos_64env"
LOG="$ROOT/logs/checkpoint_regeneration/videos_64env/render.log"
SOURCE_PATH=$(find "$ROOT/source" -mindepth 1 -maxdepth 1 -type d -printf '%p:')
export PYTHONPATH="${SOURCE_PATH}${NEWTON}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
cd "$ROOT"
exec >>"$LOG" 2>&1

ts(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log(){ echo "[$(ts)] $*"; }
fail(){ log "FAILED: $*"; exit 1; }

# Render completed runs immediately. G1 is appended after its training/validation finishes.
# id|task|experiment|last_iteration|video_steps
cases=(
  'ant|Isaac-Ant-Direct-v0|ant_direct|999|120'
  'humanoid|Isaac-Humanoid-Direct-v0|humanoid_direct|999|120'
  'go2|Isaac-Velocity-Flat-Unitree-Go2-v0|unitree_go2_flat|499|250'
  'anymal-c|Isaac-Velocity-Flat-Anymal-C-v0|anymal_c_flat|499|250'
)

render_case() {
  local spec="$1"
  IFS='|' read -r id task experiment last steps <<<"$spec"
  marker="$ROOT/logs/checkpoint_regeneration_${id}.accepted"
  [[ -s "$marker" ]] || fail "$id acceptance marker missing"
  run=$(cat "$marker")
  ckpt="$run/model_${last}.pt"
  [[ -s "$ckpt" ]] || fail "$id checkpoint missing: $ckpt"
  rawdir="$run/videos/play"
  mkdir -p "$rawdir"
  before="$OUT/.${id}.before"
  find "$rawdir" -maxdepth 1 -type f -name '*.mp4' -printf '%T@ %p\n' | sort >"$before"
  log "rendering $id: 64 envs, $steps frames, checkpoint $ckpt"
  "$PY" scripts/reinforcement_learning/rsl_rl/play.py \
    --task "$task" --headless --video --video_length "$steps" \
    --num_envs 64 --seed 20260802 --checkpoint "$ckpt" \
    presets=newton_dvi env.sim.physics=newton_dvi
  raw=$(find "$rawdir" -maxdepth 1 -type f -name '*.mp4' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  [[ -n "$raw" && -s "$raw" ]] || fail "$id produced no video"
  target="$OUT/${id}_64env_dvi_jacobi.mp4"
  cp -f "$raw" "$target"
  expected_fps=50
  [[ "$id" == ant || "$id" == humanoid ]] && expected_fps=60
  ID="$id" TARGET="$target" STEPS="$steps" EXPECTED_FPS="$expected_fps" "$PY" - <<'PY'
import cv2, os
p=os.environ['TARGET']; expected=int(os.environ['STEPS']); expected_fps=float(os.environ['EXPECTED_FPS'])
c=cv2.VideoCapture(p)
frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); fps=c.get(cv2.CAP_PROP_FPS)
w=int(c.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)); c.release()
assert (frames,w,h)==(expected,1280,720), (frames,w,h)
assert abs(fps-expected_fps)<0.1, (fps, expected_fps)
print('VALID_VIDEO',os.environ['ID'],frames,fps,w,h,p)
PY
  log "$id video accepted: $target"
}

for spec in "${cases[@]}"; do render_case "$spec"; done

log 'completed available videos; waiting for validated G1 checkpoint'
while [[ ! -s "$ROOT/logs/checkpoint_regeneration_g1.accepted" ]]; do
  if ! tmux has-session -t checkpoint-regen-supervisor 2>/dev/null; then
    fail 'checkpoint supervisor ended before G1 acceptance marker appeared'
  fi
  sleep 30
done
render_case 'g1|Isaac-Velocity-Flat-G1-v0|g1_flat|1499|250'
log 'ALL 64-ENV VIDEOS COMPLETE'
