#!/bin/bash
#SBATCH --job-name=ppo_ckpt_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_467_general
#SBATCH --mail-user=mdm9734@nyu.edu
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=ppo_ckpt_sweep_%A_%a.out
#SBATCH --error=ppo_ckpt_sweep_%A_%a.err

# Optionally train DreamerV3, then PPO fine-tune every checkpoint of it, each
# for about one checkpoint interval (scripts/ppo_finetune_checkpoints.py).
#
#   sbatch scripts/ppo_finetune_checkpoints.sh [dreamer_dir] [outdir]
#
# Runs as a job array, one task per seed (default 5 seeds: 0-4). The array
# index is used as the seed for both stages. Override the seeds at submit time,
# e.g. sbatch --array=0-9 ... or --array=3,7. Each seed gets its own directory
# outdir/seed<N>/.
#
# With dreamer_dir (a trained run's logdir containing ckpt/ and config.yaml, or
# its ckpt/ directory), stage 1 is skipped and that run is used. Without it,
# stage 1 trains DreamerV3 into outdir/seed<N>/dreamer first. PPO results go
# to outdir/seed<N>/ppo/<checkpoint name>/. A trained dreamer_dir is shared by
# all seeds (only PPO varies) unless it contains "{seed}", which is replaced by
# the seed, e.g. /scratch/mdm9734/runs/seed{seed}/dreamer. Pass "" as dreamer_dir to train while still
# giving an outdir.
#
# Environment variables (all optional; sbatch passes them through):
#   PLATFORM        jax platform: cuda or cpu (default: cuda)
#   CONFIGS         config blocks for stage 1 (default: crafter)
#   EXTRA_CONFIGS   extra config blocks appended after CONFIGS, e.g. "size12m"
#                   or "debug" for a quick smoke test
#   DREAMER_STEPS   env steps for stage 1 (default: 1.1e6)
#   SAVE_EVERY      seconds between stage 1 checkpoints (default: 900); the
#                   gap between checkpoints is what each PPO run is sized to
#   KEEP_CKPTS      how many stage 1 checkpoints to retain (default: 10; Dreamer
#                   alone keeps only 1). Each includes the replay buffer, so
#                   they are large.
#   PPO_ENVS        parallel envs, multiple of the batch size (default: 16)
#   PPO_FLAGS       extra flags for stage 2, e.g.
#                   "--agent.opt.lr 1e-5 --ppo.clip 0.1". Model flags come
#                   from the Dreamer run's config.yaml.
#
# Re-submitting with the same outdir resumes stage 2: finished checkpoints are
# skipped. Re-submit with the same dreamer_dir (the trained run) to do that
# when stage 1 is finished.
#
# e.g. sbatch scripts/ppo_finetune_checkpoints.sh   # train, then sweep
#      sbatch scripts/ppo_finetune_checkpoints.sh /scratch/mdm9734/run/dreamer

module purge

root="/scratch/mdm9734/snapbacktoreality"
dreamer="${1:-}"
seed="${SLURM_ARRAY_TASK_ID:-0}"
dreamer="${dreamer//\{seed\}/$seed}"
out="${2:-/scratch/mdm9734/ppo_ckpt_sweep/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}/seed${seed}"
platform="${PLATFORM:-cuda}"
configs="${CONFIGS:-crafter}"
extra="${EXTRA_CONFIGS:-}"
dreamer_steps="${DREAMER_STEPS:-1.1e6}"
save_every="${SAVE_EVERY:-900}"
keep_ckpts="${KEEP_CKPTS:-10}"
ppo_envs="${PPO_ENVS:-16}"
ppo_flags="${PPO_FLAGS:-}"

OVERLAY="/scratch/mdm9734/overlay_copies_gpu/dreamer-overlay-gpu.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

# Run a command inside the container with the project's python environment.
run_in_container() {
	singularity exec --nv --containall --fakeroot --cleanenv --bind=/scratch/mdm9734/trash:/tmp \
		--overlay "$OVERLAY":rw \
		"$IMAGE" \
		/bin/bash -c "source /ext3/env.sh; $*"
}

mkdir -p "$out"
echo "Seed: $seed"
echo "Output directory: $out"

if [[ -n "$dreamer" ]]; then
	echo "Using trained Dreamer run, skipping stage 1: $dreamer"
else
	dreamer="$out/dreamer"
	echo "=== Stage 1: DreamerV3 ($configs $extra, $dreamer_steps steps) ==="
	DREAMER_ARGS=(
		--configs $configs $extra
		--jax.platform "$platform"
		--batch_size 16
		--batch_length 64
		--seed "$seed"
		--logdir "$dreamer"
		--run.steps "$dreamer_steps"
		--run.save_every "$save_every"
		--run.keep_ckpts "$keep_ckpts"
		--logger.outputs jsonl
	)
	RUN_STR="$root/dreamerv3/main.py ${DREAMER_ARGS[@]}"
	echo "$RUN_STR"
	run_in_container python $RUN_STR
fi

echo "=== Stage 2: PPO fine-tune each checkpoint of $dreamer ==="
SWEEP_ARGS=(
	--dreamer "$dreamer"
	--out "$out/ppo"
	--seed "$seed"
	--jax.platform "$platform"
	--run.envs "$ppo_envs"
	--run.debug False
	--logger.outputs jsonl
	$ppo_flags
)
RUN_STR="$root/scripts/ppo_finetune_checkpoints.py ${SWEEP_ARGS[@]}"
echo "$RUN_STR"
run_in_container python $RUN_STR
