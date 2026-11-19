#!/bin/bash
# Online Dyna loop (user, 2026-11-19): act in the environment with the plan-selection planner, finetune the Dreamer 4 world
# model on what was played, then GRPO the SLM in that world model with the planner's score as the group-normalised advantage.
#
# Iteration i (aicenteritl, one GPU, every stage in its own container, sequential):
#   1 collect  eval_slm_d4_env.py: SLM_i writes 8 plans (top-20 sampling), Dreamer HEADS_i dreams 8 steps of each, score =
#              native reward + continue head + SLM value of the critic (the start SLM, fixed), the first action of the best
#              plan is played; 4 tasks x $SEEDS fresh seeds per iteration (30000000 + 1000 i ..), never the eval seeds
#              20264000..031. --save-traj keeps frames, actions, native rewards, terminals and the latents z_t.
#              Its success rate is the loop's learning curve (new layouts every iteration).
#   2 replay   build_online_d4.py: all episodes played so far, repeated to --share 0.3 of the windows, + wood2200 native.
#   3 world    train_heads.py --init HEADS_i (patch_dreamer4_heads_init.py): dynamics on z_t + reward / continue / value
#              heads, $WM_STEPS steps at lr 1e-4 / dyn_lr 2e-5 -> HEADS_{i+1}.
#   4 policy   train_slm_imagine_grpo.py --slm SLM_i --heads HEADS_{i+1} --critic SLM_0 --online <all traj dirs>
#              --tail slm: G = 8 answers per start, advantage = (score - group mean) / group std, + 0.1 BC CE on dataset
#              starts; $GRPO_UPDATES updates -> SLM_{i+1}.
# After the last iteration: the standard 128-episode wood eval (seeds 20264000..031) of SLM_N with HEADS_N, same planner.
#
#   bash dyna_loop.sh <run dir> [GPU]      env: ITERS=4 SEEDS=32 WM_STEPS=2000 GRPO_UPDATES=200 SMOKE=0
#        GRPO_ARGS="--clip 0.2 --kl-coef 0.04 --ppo-epochs 2 --minibatches 2" (KL reference = the start SLM)
set -e
D=$1; GPU=${2:-0}
ITERS=${ITERS:-4}; SEEDS=${SEEDS:-32}; WM_STEPS=${WM_STEPS:-2000}; GRPO_UPDATES=${GRPO_UPDATES:-200}; SMOKE=${SMOKE:-0}
M=$HOME/runs/slm_d4_env_eval_20261119
V=$HOME/runs/dreamer4_heads_texttask_native_wood2200_itlgpu1_20261119_r1
P=$HOME/runs/dreamer4_heads_wood2200_taskreward_itlgpu1_20261119_r1
G=$M/grpo_data
REPO=$HOME/verl-agent-craftext
IMG=gorbov-wm-verl-tf59:20261103
SLM0_RUN=$M/slm/slm5m_plan16_value_d4wood2200_r64_scratch_a3gpu2_20261118_r1
DYN=/d/dreamer4_wood2200_h200gpu1_20261107_r1/dyn/checkpoints/latest.pt
NAME=gorbov_dyna_$(basename $D | tr -c 'a-zA-Z0-9_\n' _ | cut -c1-40)
mkdir -p $D
(cd $REPO && git rev-parse --short HEAD) > $D/code_version.txt
if [ ! -d $D/dreamer4 ]; then
  cp -r $V/dreamer4 $D/dreamer4
  docker run --rm --user $(id -u):$(id -g) -v $REPO:/w:ro -v $D:/o --entrypoint python3 $IMG \
    /w/vlm_pipeline/phase4_world_model/patch_dreamer4_heads_init.py /o/dreamer4/dreamer4
fi
NV=$(cat $M/nvlibs.txt)
BASE="--gpus \"device=$GPU\" --user $(id -u):$(id -g) -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e TORCH_HOME=/o/torch_home \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -v $REPO:/w/repo:ro -v $M/mirror/dd:/d:ro -v $M/mirror/dd:/dd:ro \
  -v $P/wood2200_taskreward:$P/wood2200_taskreward:ro -v $M/mirror/c:/c:ro -v $G/latents:/c/latents:ro \
  -v $G/reason_wood2200:/c/reason_wood2200:ro -v $D/dreamer4:/code/dreamer4:ro -v $D:/D -v $M:/o"
dock() {   # dock TAG WOOD_DIR ARGS... : one stage in container ${NAME}_TAG with WOOD_DIR as the dynamics' dataset path
  local TAG=$1 WD=$2; shift 2
  eval docker run --rm --name ${NAME}_$TAG $BASE -v $WD:/d/wood2200_d4/train:ro "$@"
}

SLM_RUN=/o/slm/$(basename $SLM0_RUN); SLM_CK=step_060000.pt
CRITIC=$SLM_RUN/run/checkpoints/step_060000.pt
HEADS=$V/heads/checkpoints/latest.pt
start_it=0
for i in $(seq 0 $((ITERS - 1))); do   # resume: skip finished iterations
  [ -f $D/it$i/DONE ] || break
  SLM_RUN=/D/it$i/grpo; SLM_CK=$(cat $D/it$i/DONE); HEADS=$D/it$i/heads/checkpoints/latest.pt; start_it=$((i + 1))
done
TRAJS=""
for i in $(seq 0 $((start_it - 1))); do TRAJS="$TRAJS${TRAJS:+,}/D/it$i/traj"; done
if [ $SMOKE = 1 ]; then
  CEXTRA="--tasks 0,1 --max-steps 30"; HEXTRA="--eval_every 20 --eval_n 0 --final_eval_n 0"
  GEXTRA="--eval-every 4 --eval-starts 64 --save-every 4"
else
  CEXTRA=""; HEXTRA="--eval_every $WM_STEPS"; GEXTRA="--eval-every 100 --save-every $GRPO_UPDATES"
fi
for i in $(seq $start_it $((ITERS - 1))); do
  I=$D/it$i; mkdir -p $I
  echo "== iteration $i $(date -u) SLM $SLM_RUN/$SLM_CK HEADS $HEADS" | tee -a $D/loop.log
  s0=$((30000000 + 1000 * i))
  hd=/D/${HEADS#$D/}; [ "${HEADS#$D/}" = "$HEADS" ] && hd=/V/heads/checkpoints/latest.pt
  # 1 collect
  if [ ! -f $I/traj/z.pt ]; then
    dock collect$i $V/wood2200_native -v $V:/V:ro -w /o/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e JAX_PLATFORMS=cuda \
      -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 -e LD_LIBRARY_PATH=$NV \
      -e PYTHONPATH=/o/jaxlibs:/o/jaxcuda:/d/pylibs:/code/dreamer4 --entrypoint python3 $IMG \
      /w/repo/vlm_pipeline/phase4_world_model/eval_slm_d4_env.py --run $SLM_RUN --checkpoint $SLM_CK \
      --plans 8 --decode topk --top-k 20 --score reward --dyn $DYN --heads $hd --imagine --horizon 8 --K 1 \
      --critic $CRITIC --seeds $s0:$((s0 + SEEDS)) --seed $i --out /D/it$i/collect.json --save-traj /D/it$i/traj $CEXTRA \
      > $I/collect.log 2>&1
  fi
  python3 -c "import json; r = json.load(open('$I/collect.json')); print('collect it$i success', r['success_rate'], r['successes'], '/', r['episodes_n'], r['ends'], r['by_task'])" | tee -a $D/loop.log
  TRAJS="$TRAJS${TRAJS:+,}/D/it$i/traj"
  # 2 replay
  if [ ! -f $I/heads/checkpoints/latest.pt ]; then
    python3 $REPO/vlm_pipeline/phase4_world_model/build_online_d4.py $V/wood2200_native $I/replay $(echo $TRAJS | tr , '\n' | sed "s#^/D#$D#") --share 0.3 | tee -a $D/loop.log
    # 3 world model
    dock heads$i $I/replay -v $M/mirror/c/tok2288:/tok:ro -v $V:/V:ro -w /code/dreamer4 -e PYTHONPATH=/code/dreamer4:/d/pylibs \
      --entrypoint python3 $IMG -m dreamer4.train.train_heads --dyn $DYN --init $hd --out /D/it$i/heads \
      --steps $([ $SMOKE = 1 ] && echo 20 || echo $WM_STEPS) --lr 1e-4 --dyn_lr 2e-5 --n_tasks 4 --gamma 0.99 --text_tokenizer /tok \
      --cont_head 1 --seed $((7950 + i)) $HEXTRA > $I/heads.log 2>&1
    grep -E "HEADS-EVAL|HEADS-DONE" $I/heads.log | tail -2 | tee -a $D/loop.log
    rm -rf $I/replay
  fi
  HEADS=$I/heads/checkpoints/latest.pt
  # 4 policy
  U=$([ $SMOKE = 1 ] && echo 4 || echo $GRPO_UPDATES)
  if [ ! -f $I/grpo/checkpoints/step_$(printf %06d $U).pt ]; then
    dock grpo$i $V/wood2200_native -v $G:/g:ro -e PYTHONPATH=/code/dreamer4:/d/pylibs --entrypoint python3 $IMG \
      /w/repo/vlm_pipeline/phase4_world_model/train_slm_imagine_grpo.py --slm $SLM_RUN/run/checkpoints/$SLM_CK \
      --dyn $DYN --heads /D/it$i/heads/checkpoints/latest.pt --critic $CRITIC --online $TRAJS --online-frac 0.75 \
      --raw-data /g/raw --reasoning /c/reason_wood2200 --train /g/wood2200_v3 --val /g/wood2200_d4/val --latent-cache /c/latents \
      --out /D/it$i/grpo --updates $U --starts 32 --group 8 --tail slm --seed $((7950 + i)) $GEXTRA $GRPO_ARGS ${GRPO_ARGS:+--ref $CRITIC} \
      2>&1 | grep --line-buffered -v -iE "warn|gym" > $I/grpo.log
  fi
  grep "\[val\]" $I/grpo.log | sed -n '1p;$p' | tee -a $D/loop.log
  mkdir -p $I/grpo/run && ln -sfn ../checkpoints $I/grpo/run/checkpoints
  SLM_RUN=/D/it$i/grpo; SLM_CK=step_$(printf %06d $U).pt
  echo $SLM_CK > $I/DONE
done
# final: the standard eval seeds with the last SLM and world model
if [ $SMOKE != 1 ] && [ ! -f $D/final_eval.json ]; then
  hd=/D/${HEADS#$D/}
  dock final $V/wood2200_native -v $V:/V:ro -w /o/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e JAX_PLATFORMS=cuda \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 -e LD_LIBRARY_PATH=$NV \
    -e PYTHONPATH=/o/jaxlibs:/o/jaxcuda:/d/pylibs:/code/dreamer4 --entrypoint python3 $IMG \
    /w/repo/vlm_pipeline/phase4_world_model/eval_slm_d4_env.py --run $SLM_RUN --checkpoint $SLM_CK \
    --plans 8 --decode topk --top-k 20 --score reward --dyn $DYN --heads $hd --imagine --horizon 8 --K 1 \
    --critic $CRITIC --seed 0 --out /D/final_eval.json > $D/final_eval.log 2>&1
  python3 -c "import json; r = json.load(open('$D/final_eval.json')); print('FINAL eval seeds success', r['success_rate'], r['successes'], r['ends'], r['by_task'])" | tee -a $D/loop.log
fi
echo "DYNA_DONE $(date -u)" | tee -a $D/loop.log
