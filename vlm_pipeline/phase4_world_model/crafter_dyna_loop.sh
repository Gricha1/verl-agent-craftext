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
#        TOK_FT=0 TOK_STEPS=3000 TOK_ANCHOR=10 TOK_LPIPS=0.2 DYN_LR=2e-5: TOK_FT=1 (user 2026-11-20) finetunes the tokenizer's
#        encoder + decoder after each replay (finetune_d4_tokenizer.py, see dyna_loop.sh 2b); GRPO re-encodes only the played
#        episodes (the 3M Qwen frames keep their cached latents, the anchor holds them)
#        GRPO_ARGS="--clip 0.2 --kl-coef 0.04 --ppo-epochs 2 --minibatches 2" (KL reference = the start SLM)
#   3c critic  CRITIC_HL=<HL-Gauss critic .pt, container path> (user 2026-11-21, the ZeroHero critic): the stage-1 critic of
#              train_slm_reason_act_d4.py --critic-from replaces the start SLM's value everywhere (base eval, collect, GRPO
#              tail, final eval); after the world model, finetune_slm_critic_real.py moves it to the GAE(0.99, 0.95) returns of
#              the planner's real episodes (all traj dirs so far, 25 % teacher frames as an anchor), $CRIT_STEPS steps, from the
#              previous iteration's critic; GRPO and the next collect use it. With GRPO_ARGS="--real-weight 1 ..." GRPO also
#              distills the planner's real choices weighted by their real advantage (train_slm_imagine_grpo.py doc).
#        CRITIC_HL="" CRIT_STEPS=1500
#   3d dream   DREAM_HEADS=1 (user 2026-11-22): train_heads_on_dreams.py trains the reward / continue heads and the critic on
#              dreamed windows of real actions (teacher data + every played episode) against the real outcomes, from the heads
#              of step 3 and the current critic; GRPO and the next collect score plans with them. DH0_HEADS / DH0_CRITIC: the
#              Stage-1 dream-trained scorer used from the start. EVAL_EXTRA: extra planner args of the evals.
#        DREAM_HEADS=0 DH_STEPS=1500 DH0_HEADS="" DH0_CRITIC="" EVAL_EXTRA=""
#   state      STATE_CARRY=1 (default since 2026-11-22, safe_rl_nlp.md 14.15-17): every trainer of iteration i>0 restores
#              the training state of the same trainer in iteration i-1 (--state-in .../train_state.pt: AdamW moments and step,
#              the global step of its lr schedule, the RMS loss normalisers) and its lr schedule runs over ITERS x its steps,
#              so the 4 iterations are one continuous optimisation per model. Before, every iteration restarted AdamW, the
#              warmup + cosine and the normalisers. Iteration 0 starts fresh (new phase after Stage 1). A missing state of
#              iteration i-1 stops the loop (never continue without the full state).
set -eo pipefail   # a failed stage must stop the loop (its output is piped through grep / tee)
D=$1; S0=$2; CK0=$3; GPU=${4:-3}
ITERS=${ITERS:-4}; SEEDS=${SEEDS:-64}; CSTEPS=${CSTEPS:-1000}; WM_STEPS=${WM_STEPS:-2000}; GRPO_UPDATES=${GRPO_UPDATES:-200}
BASE_EPS=${BASE_EPS:-3000}; EVAL_SEEDS=${EVAL_SEEDS:-128}; EVAL_STEPS=${EVAL_STEPS:-2000}; SMOKE=${SMOKE:-0}
CRITIC_HL=${CRITIC_HL:-}; CRIT_STEPS=${CRIT_STEPS:-1500}
DREAM_HEADS=${DREAM_HEADS:-0}; DH_STEPS=${DH_STEPS:-1500}; DH0_HEADS=${DH0_HEADS:-}; DH0_CRITIC=${DH0_CRITIC:-}; EVAL_EXTRA=${EVAL_EXTRA:-}
TOK_FT=${TOK_FT:-0}; TOK_STEPS=${TOK_STEPS:-3000}; TOK_ANCHOR=${TOK_ANCHOR:-10}; TOK_LPIPS=${TOK_LPIPS:-0.2}; DYN_LR=${DYN_LR:-2e-5}
B=/mnt/aicenter1-datasets/gorbov_gv/modern_verl_reasoning_rl_a3_20261016
REPO=${REPO:-$B/repo/verl-agent-craftext}   # override: a second checkout while a loop runs from this one
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
  for P in patch_dreamer4_heads_text_task.py patch_dreamer4_heads_continue.py patch_dreamer4_heads_init.py patch_dreamer4_heads_state.py; do   # continue needs text_task
    CUDA_VISIBLE_DEVICES=$GPU docker run --rm --user 1007:100 -v $REPO:/w:ro -v $D:/o --entrypoint python3 $IMG \
      /w/vlm_pipeline/phase4_world_model/$P /o/dreamer4/dreamer4
  done
  touch $D/dreamer4/PATCHED
fi
DYN0=/r/dyn/checkpoints/latest.pt; DYN=$DYN0; TOK=""   # TOK empty: the phase-1 tokenizer (the SLM's own)
CRITIC=${CRITIC_HL:-/s0/run/checkpoints/$CK0}   # the stage-1 critic: the start SLM's value, or the HL-Gauss critic
[ -n "$DH0_CRITIC" ] && CRITIC=$DH0_CRITIC        # the Stage-1 critic trained on dreams
[ -n "$DH0_HEADS" ] && H0=$DH0_HEADS              # the Stage-1 heads trained on dreams
PLAN="--game crafter --plans 8 --decode topk --top-k 20 --score reward --dyn $DYN --imagine --horizon 8 --K 1 --critic $CRITIC"
EV="--entrypoint python3 $IMG /w/repo/vlm_pipeline/phase4_world_model/eval_slm_d4_env.py"
if [ $SMOKE = 1 ]; then EVAL_SEEDS=4; EVAL_STEPS=20; SEEDS=4; CSTEPS=30; fi
E0=150000000; E1=$((E0 + EVAL_SEEDS))
# 0 base
if [ ! -f $D/base_eval.json ]; then
  echo "== base eval $(date -u) SLM $S0/$CK0 HEADS $H0" | tee -a $D/loop.log
  dock base $Q/crafter64/train $EV --run /s0 --checkpoint $CK0 --heads /runs/${H0#$B/run/} $PLAN $EVAL_EXTRA --seeds $E0:$E1 \
    --max-steps $EVAL_STEPS --seed 0 --out /D/base_eval.json > $D/base_eval.log 2>&1
fi
python3 -c "import json; r = json.load(open('$D/base_eval.json')); print('BASE eval score', round(r['score'], 3), 'return', round(r['mean_return'], 2), r['ends'], {k: round(v, 2) for k, v in r['achievement_rates'].items()})" | tee -a $D/loop.log

SLM_RUN=/s0; SLM_CK=$CK0; HEADS=$H0
start_it=0
for i in $(seq 0 $((ITERS - 1))); do   # resume
  [ -f $D/it$i/DONE ] || break
  SLM_RUN=/D/it$i/grpo; SLM_CK=$(cat $D/it$i/DONE); HEADS=$D/it$i/heads/checkpoints/latest.pt; start_it=$((i + 1))
  [ -f $D/it$i/tok/dyn_tok.pt ] && { DYN=/D/it$i/tok/dyn_tok.pt; TOK=/D/it$i/tok/latest.pt; }
  [ -f $D/it$i/critic/critic.pt ] && CRITIC=/D/it$i/critic/critic.pt
  [ -f $D/it$i/dheads/heads.pt ] && { HEADS=$D/it$i/dheads/heads.pt; CRITIC=/D/it$i/dheads/critic.pt; }
done
TRAJS=""
for i in $(seq 0 $((start_it - 1))); do TRAJS="$TRAJS${TRAJS:+,}/D/it$i/traj"; done
if [ $SMOKE = 1 ]; then
  HEXTRA="--eval_every 20 --eval_n 0 --final_eval_n 0"; TOK_STEPS=20; CRIT_STEPS=20; GEXTRA="--eval-every 4 --eval-starts 64 --save-every 4"
else
  HEXTRA="--eval_every $WM_STEPS"; GEXTRA="--eval-every 100 --save-every $GRPO_UPDATES"
fi
hpath() { case $1 in $D/*) echo /D/${1#$D/};; *) echo /runs/${1#$B/run/};; esac; }
STATE_CARRY=${STATE_CARRY:-1}
chk() {   # chk SUBPATH: the state of iteration i-1 must exist (iteration 0 starts fresh)
  if [ "$STATE_CARRY" = 1 ] && [ $i -gt 0 ] && [ ! -f $D/it$((i - 1))/$1 ]; then
    echo "MISSING TRAIN STATE $D/it$((i - 1))/$1 (safe_rl_nlp.md 14.17): not continuing without it" | tee -a $D/loop.log; exit 1
  fi
}
st() { if [ "$STATE_CARRY" = 1 ] && [ $i -gt 0 ]; then echo "$1 /D/it$((i - 1))/$2"; fi; }   # st FLAG SUBPATH
for i in $(seq $start_it $((ITERS - 1))); do
  I=$D/it$i; mkdir -p $I
  echo "== iteration $i $(date -u) SLM $SLM_RUN/$SLM_CK HEADS $HEADS DYN $DYN TOK ${TOK:-phase1}" | tee -a $D/loop.log
  s0=$((170000000 + 1000 * i)); hd=$(hpath $HEADS)
  # 1 collect
  if [ ! -f $I/traj/z.pt ]; then
    dock collect$i $Q/crafter64/train $EV --run $SLM_RUN --checkpoint $SLM_CK --heads $hd $PLAN --dyn $DYN ${TOK:+--tok $TOK} --critic $CRITIC $EVAL_EXTRA --seeds $s0:$((s0 + SEEDS)) \
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
    if [ $TOK_FT = 1 ] && [ ! -f $I/tok/dyn_tok.pt ]; then   # 2b tokenizer: encoder + decoder on the played frames
      chk tok/train_state.pt
      dock tok$i $Q/crafter64/train --entrypoint python3 $IMG /w/repo/vlm_pipeline/phase4_world_model/finetune_d4_tokenizer.py \
        ${TOK:+--tok $TOK} --dyn $DYN0 --base /qd/train --online $TRAJS --out /D/it$i/tok --steps $TOK_STEPS $(st --state-in tok/train_state.pt) --total-steps $((ITERS * TOK_STEPS)) --anchor $TOK_ANCHOR \
        --perceptual $TOK_LPIPS --seed $((7950 + i)) > $I/tok.log 2>&1
      grep -E "TOK-FT (before|after)" $I/tok.log | tee -a $D/loop.log
    fi
    if [ $TOK_FT = 1 ]; then DYN=/D/it$i/tok/dyn_tok.pt; TOK=/D/it$i/tok/latest.pt; fi
    chk heads/checkpoints/train_state.pt
    dock heads$i $I/replay -w /code/dreamer4 --entrypoint python3 $IMG -m dreamer4.train.train_heads --dyn $DYN --init $hd $(st --state_in heads/checkpoints/train_state.pt) --total_steps $((ITERS * $([ $SMOKE = 1 ] && echo 20 || echo $WM_STEPS))) \
      --out /D/it$i/heads --steps $([ $SMOKE = 1 ] && echo 20 || echo $WM_STEPS) --lr 1e-4 --dyn_lr $DYN_LR --n_tasks 1 \
      --cont_head 1 --seed $((7950 + i)) $HEXTRA > $I/heads.log 2>&1
    grep -E "init|HEADS-EVAL|HEADS-DONE" $I/heads.log | tail -3 | tee -a $D/loop.log
    rm -rf $I/replay
  fi
  HEADS=$I/heads/checkpoints/latest.pt
  if [ $TOK_FT = 1 ]; then DYN=/D/it$i/tok/dyn_tok.pt; TOK=/D/it$i/tok/latest.pt; fi   # (a resumed iteration skipped 2b)
  # 3c critic: the HL-Gauss critic on the real returns of the planner
  REAL_CRITIC=""
  if [ -n "$CRITIC_HL" ]; then
    REAL_CRITIC=$CRITIC   # the critic these episodes were played with: V_old of the real advantages
    if [ ! -f $I/critic/critic.pt ]; then
      chk critic/train_state.pt
      dock critic$i $Q/crafter64/train --entrypoint python3 $IMG /w/repo/vlm_pipeline/phase4_world_model/finetune_slm_critic_real.py \
        --critic $CRITIC --online $TRAJS --game crafter --train /qd/train --val /qd/val --latent-cache /s0/latents \
        --out /D/it$i/critic --steps $CRIT_STEPS $(st --state-in critic/train_state.pt) --total-steps $((ITERS * CRIT_STEPS)) --seed $((7950 + i)) > $I/critic.log 2>&1
    fi
    grep -E "CRITIC-FT (before|after)" $I/critic.log | tee -a $D/loop.log
    CRITIC=/D/it$i/critic/critic.pt
  fi
  # 3d the scorer on dreams of real actions (teacher data + all played episodes)
  if [ "$DREAM_HEADS" = 1 ]; then
    if [ ! -f $I/dheads/heads.pt ]; then
      chk dheads/train_state.pt
      dock dheads$i $Q/crafter64/train --entrypoint python3 $IMG /w/repo/vlm_pipeline/phase4_world_model/train_heads_on_dreams.py \
        --dyn $DYN --heads /D/it$i/heads/checkpoints/latest.pt --critic $CRITIC --game crafter --data /qd/train \
        --latents /s0/latents/z_train.pt --online $TRAJS --out /D/it$i/dheads $(st --state-in dheads/train_state.pt) --total-steps $((ITERS * $([ $SMOKE = 1 ] && echo 20 || echo $DH_STEPS))) --steps $([ $SMOKE = 1 ] && echo 20 || echo $DH_STEPS) \
        $([ $SMOKE = 1 ] && echo --eval-batches 2) --seed $((7950 + i)) > $I/dheads.log 2>&1
    fi
    grep -E "DREAM-HEADS (before|after)" $I/dheads.log | tee -a $D/loop.log
    HEADS=$I/dheads/heads.pt; CRITIC=/D/it$i/dheads/critic.pt
  fi
  # 4 policy
  U=$([ $SMOKE = 1 ] && echo 4 || echo $GRPO_UPDATES)
  if [ ! -f $I/grpo/checkpoints/step_$(printf %06d $U).pt ]; then
    chk grpo/train_state.pt
    dock grpo$i $Q/crafter64/train --entrypoint python3 $IMG /w/repo/vlm_pipeline/phase4_world_model/train_slm_imagine_grpo.py \
      --slm $SLM_RUN/run/checkpoints/$SLM_CK --dyn $DYN $([ $TOK_FT = 1 ] && echo --reencode online) --heads $(hpath $HEADS) --critic $CRITIC \
      --online $TRAJS --online-frac 0.75 --frame-labels /ql --train /qd/train --val /qd/val --latent-cache /s0/latents \
      --out /D/it$i/grpo $(st --state-in grpo/train_state.pt) --updates $U --starts 32 --group 8 --tail slm --seed $((7950 + i)) $GEXTRA $GRPO_ARGS ${REAL_CRITIC:+--real-critic $REAL_CRITIC} ${GRPO_ARGS:+--ref /s0/run/checkpoints/$CK0} \
      2>&1 | grep --line-buffered -v -iE "warn|gym|numpy 2|upgrade to|migration" > $I/grpo.log
  fi
  grep "\[val\]" $I/grpo.log | sed -n '1p;$p' | tee -a $D/loop.log
  mkdir -p $I/grpo/run && ln -sfn ../checkpoints $I/grpo/run/checkpoints
  SLM_RUN=/D/it$i/grpo; SLM_CK=step_$(printf %06d $U).pt
  echo $SLM_CK > $I/DONE
done
# final: the base eval with the last SLM and world model
if [ ! -f $D/final_eval.json ]; then
  dock final $Q/crafter64/train $EV --run $SLM_RUN --checkpoint $SLM_CK --heads $(hpath $HEADS) $PLAN --dyn $DYN ${TOK:+--tok $TOK} --critic $CRITIC $EVAL_EXTRA --seeds $E0:$E1 \
    --max-steps $EVAL_STEPS --seed 0 --out /D/final_eval.json > $D/final_eval.log 2>&1
fi
python3 -c "import json; r = json.load(open('$D/final_eval.json')); print('FINAL eval score', round(r['score'], 3), 'return', round(r['mean_return'], 2), r['ends'], {k: round(v, 2) for k, v in r['achievement_rates'].items()})" | tee -a $D/loop.log
# keep only the final weights (user 2026-11-20: "промежуточные веса удаляй ток финальные оставляй"): after the final eval the
# checkpoints of every iteration but the last go, and the re-encoded latent caches of all iterations; traj z.pt stay
if [ -f $D/final_eval.json ]; then
  for j in $(seq 0 $((ITERS - 2))); do find $D/it$j -type f \( -name '*.pt' -o -name '*.pth' \) ! -name 'z*.pt' -delete 2>/dev/null; done
  rm -rf $D/it*/grpo/latents 2>/dev/null
  for j in $(seq 0 $((ITERS - 2))); do rm -f $D/it$j/critic/critic.pt; done
  echo "intermediate weights deleted, kept it$((ITERS - 1))" | tee -a $D/loop.log
fi
echo "CDYNA_DONE $(date -u)" | tee -a $D/loop.log
