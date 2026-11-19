#!/bin/bash
# Phase 2 (RL with the environment) of the Qwen-teacher Crafter pipeline (Alexey's data, user 2026-11-19): the same online Dyna
# loop as dyna_loop.sh (wood), in stock Crafter 1.8.3 on aicenter3. Phase 1 (BC) = the Dreamer 4 of the Qwen data
# (dreamer4_qwen_crafter64_a3gpu3_20261118_r1 + phase-2 heads dreamer4_heads_qwen_crafter64_reward_a3gpu3_20261119_r1) and the
# SLM plan16 + value of train_slm_reason_act_d4.py --frame-labels.
#
#   0 base     the start SLM and heads with the planner on the ZeroHero eval worlds (seeds 150000000..127, 2000 steps):
#              the number the loop has to beat (score = distinct achievements per episode, unlock rate per achievement)
# Iteration i (one GPU, each stage in its own container):
#   1 collect  eval_slm_d4_env.py --game crafter: SLM_i writes 8 plans (top-20 sampling), Dreamer HEADS_i dreams 8 steps of
#              each, score = native reward + continue head (from iteration 1) + value of the critic (the start SLM, fixed);
#              the first action of the best plan is played; $SEEDS fresh worlds (170000000 + 1000 i ..), $CSTEPS steps at most
#   2 replay   build_online_d4.py: all episodes played so far, repeated to --share 0.3, + $BASE_EPS random Qwen episodes
#   3 world    train_heads.py --init HEADS_i --cont_head 1 (a continue head on the recorded deaths, new at iteration 0)
#   4 policy   train_slm_imagine_grpo.py --frame-labels: GRPO in the world model, advantage = group-normalised plan score,
#              75 % of the starts from the played episodes, BC anchor 0.1 on Qwen-labelled dataset starts
# Final: the base eval again with SLM_N and HEADS_N.
#
#   bash crafter_dyna_loop.sh <run dir> <SLM run dir> <SLM checkpoint> [GPU]
#   env: ITERS=4 SEEDS=64 CSTEPS=1000 WM_STEPS=2000 GRPO_UPDATES=200 BASE_EPS=3000 EVAL_SEEDS=128 EVAL_STEPS=2000 SMOKE=0
set -e
D=$1; S0=$2; CK0=$3; GPU=${4:-3}
ITERS=${ITERS:-4}; SEEDS=${SEEDS:-64}; CSTEPS=${CSTEPS:-1000}; WM_STEPS=${WM_STEPS:-2000}; GRPO_UPDATES=${GRPO_UPDATES:-200}
BASE_EPS=${BASE_EPS:-3000}; EVAL_SEEDS=${EVAL_SEEDS:-128}; EVAL_STEPS=${EVAL_STEPS:-2000}; SMOKE=${SMOKE:-0}
B=/mnt/aicenter1-datasets/gorbov_gv/modern_verl_reasoning_rl_a3_20261016
REPO=$B/repo/verl-agent-craftext
Q=$B/data/zerohero_crafter_memory_20261106
Q4=$B/run/dreamer4_qwen_crafter64_a3gpu3_20261118_r1
H0=$B/run/dreamer4_heads_qwen_crafter64_reward_a3gpu3_20261119_r1/heads/checkpoints/latest.pt
CODE=$B/run/dreamer4_heads_code_20261119/dreamer4
IMG=modern-verl-craftext:v091-smoke-craftext
NAME=gorbov_cdyna_$(basename $D | tr -c 'a-zA-Z0-9_\n' _ | cut -c1-40)
mkdir -p $D
(cd $REPO && git rev-parse --short HEAD) > $D/code_version.txt
printf 'USER=gorbov_gv\nLOGNAME=gorbov_gv\nHOME=/tmp\nPYTHONUNBUFFERED=1\nPYTHONPATH=/cl:/p/pylibs:/code/dreamer4\nTORCH_HOME=/c/torch_home\nPYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\nNUMBA_CACHE_DIR=/tmp/numba\n' > $D/torch.env
dock() {   # dock TAG DYN_DATA_DIR ARGS...: one stage in container ${NAME}_TAG (the aicenter3 wrapper pins $GPU, drops -e)
  local TAG=$1 WD=$2; shift 2
  CUDA_VISIBLE_DEVICES=$GPU docker run --rm --name ${NAME}_$TAG --user 1007:100 --env-file $D/torch.env \
    -v $REPO:/w/repo:ro -v $B/run/dreamer_eval_wood2200_a3gpu3_20261112_r1/pylibs:/p/pylibs:ro -v $B/run/crafter_dyna_libs_20261119:/cl:ro \
    -v $Q4:/q4:ro -v $Q4:/r:ro -v $B/run/slm5m_recon_d4wood2200_init_20261118:/c:ro -v $Q/crafter64:/qd:ro -v $WD:/q/train:ro \
    -v $Q/slm_labels:/ql:ro -v $S0:/s0:ro -v $D/dreamer4:/code/dreamer4:ro -v $B/run:/runs:ro -v $D:/D "$@"
}
if [ ! -f $D/dreamer4/PATCHED ]; then   # dreamer4 of the Qwen heads + task-text (unused) + continue head + --init
  rm -rf $D/dreamer4; cp -r $CODE $D/dreamer4
  for P in patch_dreamer4_heads_text_task.py patch_dreamer4_heads_continue.py patch_dreamer4_heads_init.py; do   # continue needs text_task
    CUDA_VISIBLE_DEVICES=$GPU docker run --rm --user 1007:100 -v $REPO:/w:ro -v $D:/o --entrypoint python3 $IMG \
      /w/vlm_pipeline/phase4_world_model/$P /o/dreamer4/dreamer4
  done
  touch $D/dreamer4/PATCHED
fi
DYN=/r/dyn/checkpoints/latest.pt
CRITIC=/s0/run/checkpoints/$CK0
PLAN="--game crafter --plans 8 --decode topk --top-k 20 --score reward --dyn $DYN --imagine --horizon 8 --K 1 --critic $CRITIC"
EV="--entrypoint python3 $IMG /w/repo/vlm_pipeline/phase4_world_model/eval_slm_d4_env.py"
if [ $SMOKE = 1 ]; then EVAL_SEEDS=4; EVAL_STEPS=20; SEEDS=4; CSTEPS=30; fi
E0=150000000; E1=$((E0 + EVAL_SEEDS))
# 0 base
if [ ! -f $D/base_eval.json ]; then
  echo "== base eval $(date -u) SLM $S0/$CK0 HEADS $H0" | tee -a $D/loop.log
  dock base $Q/crafter64/train $EV --run /s0 --checkpoint $CK0 --heads /runs/${H0#$B/run/} $PLAN --seeds $E0:$E1 \
    --max-steps $EVAL_STEPS --seed 0 --out /D/base_eval.json > $D/base_eval.log 2>&1
fi
python3 -c "import json; r = json.load(open('$D/base_eval.json')); print('BASE eval score', round(r['score'], 3), 'return', round(r['mean_return'], 2), r['ends'], {k: round(v, 2) for k, v in r['achievement_rates'].items()})" | tee -a $D/loop.log

SLM_RUN=/s0; SLM_CK=$CK0; HEADS=$H0
start_it=0
for i in $(seq 0 $((ITERS - 1))); do   # resume
  [ -f $D/it$i/DONE ] || break
  SLM_RUN=/D/it$i/grpo; SLM_CK=$(cat $D/it$i/DONE); HEADS=$D/it$i/heads/checkpoints/latest.pt; start_it=$((i + 1))
done
TRAJS=""
for i in $(seq 0 $((start_it - 1))); do TRAJS="$TRAJS${TRAJS:+,}/D/it$i/traj"; done
if [ $SMOKE = 1 ]; then
  HEXTRA="--eval_every 20 --eval_n 0 --final_eval_n 0"; GEXTRA="--eval-every 4 --eval-starts 64 --save-every 4"
else
  HEXTRA="--eval_every $WM_STEPS"; GEXTRA="--eval-every 100 --save-every $GRPO_UPDATES"
fi
hpath() { case $1 in $D/*) echo /D/${1#$D/};; *) echo /runs/${1#$B/run/};; esac; }
for i in $(seq $start_it $((ITERS - 1))); do
  I=$D/it$i; mkdir -p $I
  echo "== iteration $i $(date -u) SLM $SLM_RUN/$SLM_CK HEADS $HEADS" | tee -a $D/loop.log
  s0=$((170000000 + 1000 * i)); hd=$(hpath $HEADS)
  # 1 collect
  if [ ! -f $I/traj/z.pt ]; then
    dock collect$i $Q/crafter64/train $EV --run $SLM_RUN --checkpoint $SLM_CK --heads $hd $PLAN --seeds $s0:$((s0 + SEEDS)) \
      --max-steps $CSTEPS --seed $((i + 1)) --out /D/it$i/collect.json --save-traj /D/it$i/traj > $I/collect.log 2>&1
  fi
  python3 -c "import json; r = json.load(open('$I/collect.json')); print('collect it$i score', round(r['score'], 3), 'return', round(r['mean_return'], 2), r['ends'], 'steps', round(r['mean_steps'], 1))" | tee -a $D/loop.log
  TRAJS="$TRAJS${TRAJS:+,}/D/it$i/traj"
  # 2 replay + 3 world model
  if [ ! -f $I/heads/checkpoints/latest.pt ]; then
    mkdir -p $I/replay
    dock replay$i $Q/crafter64/train --entrypoint python3 $IMG /w/repo/vlm_pipeline/phase4_world_model/build_online_d4.py \
      /qd/train /D/it$i/replay $(echo $TRAJS | tr , ' ') --share 0.3 --base-episodes $([ $SMOKE = 1 ] && echo 40 || echo $BASE_EPS) \
      --seed $i | tee -a $D/loop.log
    dock heads$i $I/replay -w /code/dreamer4 --entrypoint python3 $IMG -m dreamer4.train.train_heads --dyn $DYN --init $hd \
      --out /D/it$i/heads --steps $([ $SMOKE = 1 ] && echo 20 || echo $WM_STEPS) --lr 1e-4 --dyn_lr 2e-5 --n_tasks 1 \
      --cont_head 1 --seed $((7950 + i)) $HEXTRA > $I/heads.log 2>&1
    grep -E "init|HEADS-EVAL|HEADS-DONE" $I/heads.log | tail -3 | tee -a $D/loop.log
    rm -rf $I/replay
  fi
  HEADS=$I/heads/checkpoints/latest.pt
  # 4 policy
  U=$([ $SMOKE = 1 ] && echo 4 || echo $GRPO_UPDATES)
  if [ ! -f $I/grpo/checkpoints/step_$(printf %06d $U).pt ]; then
    dock grpo$i $Q/crafter64/train --entrypoint python3 $IMG /w/repo/vlm_pipeline/phase4_world_model/train_slm_imagine_grpo.py \
      --slm $SLM_RUN/run/checkpoints/$SLM_CK --dyn $DYN --heads /D/it$i/heads/checkpoints/latest.pt --critic $CRITIC \
      --online $TRAJS --online-frac 0.75 --frame-labels /ql --train /qd/train --val /qd/val --latent-cache /s0/latents \
      --out /D/it$i/grpo --updates $U --starts 32 --group 8 --tail slm --seed $((7950 + i)) $GEXTRA \
      2>&1 | grep --line-buffered -v -iE "warn|gym|numpy 2|upgrade to|migration" > $I/grpo.log
  fi
  grep "\[val\]" $I/grpo.log | sed -n '1p;$p' | tee -a $D/loop.log
  mkdir -p $I/grpo/run && ln -sfn ../checkpoints $I/grpo/run/checkpoints
  SLM_RUN=/D/it$i/grpo; SLM_CK=step_$(printf %06d $U).pt
  echo $SLM_CK > $I/DONE
done
# final: the base eval with the last SLM and world model
if [ ! -f $D/final_eval.json ]; then
  dock final $Q/crafter64/train $EV --run $SLM_RUN --checkpoint $SLM_CK --heads $(hpath $HEADS) $PLAN --seeds $E0:$E1 \
    --max-steps $EVAL_STEPS --seed 0 --out /D/final_eval.json > $D/final_eval.log 2>&1
fi
python3 -c "import json; r = json.load(open('$D/final_eval.json')); print('FINAL eval score', round(r['score'], 3), 'return', round(r['mean_return'], 2), r['ends'], {k: round(v, 2) for k, v in r['achievement_rates'].items()})" | tee -a $D/loop.log
echo "CDYNA_DONE $(date -u)" | tee -a $D/loop.log
