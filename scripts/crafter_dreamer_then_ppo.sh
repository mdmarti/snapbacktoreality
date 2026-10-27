#!/bin/bash
#SBATCH --job-name=crafter_dreamer_ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_467_general
#SBATCH --mail-user=mdm9734@nyu.edu
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:1
#SBATCH --output=crafter_dreamer_ppo_%j.out
#SBATCH --error=crafter_dreamer_ppo_%j.err

# Train DreamerV3 on Crafter with the repo's hyperparameters, then freeze it and
# fine-tune only the policy with PPO on real rewards (dreamerv3/ppo_finetune.py).
#
#   sbatch scripts/crafter_dreamer_then_ppo.sh [outdir]
#
# Environment variables (all optional; sbatch passes them through):
#   PLATFORM        jax platform: cuda or cpu (default: cuda)
#   EXTRA_CONFIGS   extra config blocks appended after `crafter`, e.g. "size12m"
#                   or "debug" for a quick smoke test
#   DREAMER_STEPS   env steps for stage 1 (default: 1.1e6, the crafter config)
#   PPO_ITERS       PPO iterations for stage 2 (default: 500; each iteration is
#                   PPO_ENVS * 64 env steps, so 500 iterations ~ 512k steps)
#   PPO_ENVS        parallel envs for stage 2, multiple of 16 (default: 16)
#   PPO_FLAGS       extra flags for stage 2 only, e.g.
#                   "--agent.opt.warmup 50 --agent.opt.lr 1e-5 --ppo.clip 0.1"
#                   (the model flags must stay identical between stages, so
#                   only pass optimizer/PPO settings here)
#   SAVE_EVERY      seconds between stage 1 checkpoints (default: 900; each one
#                   also writes the replay buffer, so lower is slower)
#   SKIP_DREAMER=1  reuse an existing outdir/dreamer run and only do stage 2
#   SKIP_PPO=1      only do stage 1 (Dreamer); skips PPO and the comparison
#
# e.g. DREAMER_STEPS=2e6 PPO_ITERS=1000 sbatch scripts/crafter_dreamer_then_ppo.sh

module purge

root="/scratch/mdm9734/snapbacktoreality"
out="${1:-/scratch/mdm9734/crafter_dreamer_ppo/${SLURM_JOB_ID}}"
platform="${PLATFORM:-cuda}"
extra="${EXTRA_CONFIGS:-}"
dreamer_steps="${DREAMER_STEPS:-1.1e6}"
ppo_iters="${PPO_ITERS:-500}"
ppo_envs="${PPO_ENVS:-16}"
ppo_flags="${PPO_FLAGS:-}"
save_every="${SAVE_EVERY:-900}"

OVERLAY="/scratch/mdm9734/overlay_copies_gpu/dreamer-overlay-gpu.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

# Run a command inside the container with the project's python environment.
run_in_container() {
	singularity exec --nv --containall --fakeroot --cleanenv --bind=/scratch/mdm9734/trash:/tmp \
		--overlay "$OVERLAY":rw \
		"$IMAGE" \
		/bin/bash -c "source /ext3/env.sh; $*"
}

# Both stages must build the identical model, so they share these flags.
MODEL_ARGS=(
	--configs crafter $extra
	--jax.platform "$platform"
	--batch_size 16
	--batch_length 64
)

mkdir -p "$out"
echo "Output directory: $out"

if [[ -z "${SKIP_DREAMER:-}" ]]; then
	echo "=== Stage 1: DreamerV3 on Crafter ($dreamer_steps steps) ==="
	DREAMER_ARGS=(
		"${MODEL_ARGS[@]}"
		--logdir "$out/dreamer"
		--run.steps "$dreamer_steps"
		--run.save_every "$save_every"
		--logger.outputs jsonl
	)
	RUN_STR="$root/dreamerv3/main.py ${DREAMER_ARGS[@]}"
	echo "$RUN_STR"
	run_in_container python $RUN_STR
fi

if [[ -n "${SKIP_PPO:-}" ]]; then
	echo "SKIP_PPO set: skipping stage 2 and the comparison"
	exit 0
fi

echo "=== Stage 2: freeze world model/value, PPO on the policy ==="
PPO_ARGS=(
	"${MODEL_ARGS[@]}"
	--ckpt "$out/dreamer"
	--logdir "$out/ppo"
	--run.envs "$ppo_envs"
	--run.debug False
	--ppo.iters "$ppo_iters"
	--logger.outputs jsonl
	$ppo_flags
)
RUN_STR="$root/dreamerv3/ppo_finetune.py ${PPO_ARGS[@]}"
echo "$RUN_STR"
run_in_container python $RUN_STR

echo "=== Comparison (mean Crafter score over the last 100 episodes) ==="
cat > "$out/compare.py" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
for name in ('dreamer', 'ppo'):
  path = out / name / 'scores.jsonl'
  scores = [json.loads(l)['episode/score'] for l in path.open() if l.strip()]
  tail = scores[-100:]
  mean = sum(tail) / max(len(tail), 1)
  print(f'{name:8s} episodes={len(scores):6d}  last-100 mean score={mean:.2f}')
PY
run_in_container python -I "$out/compare.py" "$out"
