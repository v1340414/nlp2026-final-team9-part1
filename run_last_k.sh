#!/usr/bin/env bash
# Last-k layer fine-tuning 실험 (gpt2-xl).
# 데이터: train=sonnets.txt, dev프롬프트=sonnets_held_out_dev.txt, dev정답=TRUE_sonnets_held_out_dev.txt
#        (test 소넷은 사용하지 않음)
# 각 실험이 끝나면 [RESULT] 블록으로 모든 지표가 터미널에 바로 출력됨.
set -e

MODEL=gpt2-xl
EPOCHS=10
BS=4              # gpt2-xl 메모리 큼: 4부터 시작, OOM이면 2
SEED=11711
TEMP=0.7
TOPP=0.85
MAXLEN=180

run () {  # run <tag> <last_k> <lr>
  local TAG=$1 LK=$2 LR=$3
  echo "===== ${TAG}: last_k=${LK}, lr=${LR} ====="
  python sonnet_generation.py \
    --use_gpu \
    --model_size ${MODEL} \
    --last_k ${LK} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --batch_size ${BS} \
    --seed ${SEED} \
    --temperature ${TEMP} \
    --top_p ${TOPP} \
    --max_length ${MAXLEN} \
    --sonnet_path data/sonnets.txt \
    --held_out_sonnet_path data/sonnets_held_out_dev.txt \
    --gold_path data/TRUE_sonnets_held_out_dev.txt \
    --sonnet_out predictions/${MODEL}-${TAG}-sonnets.txt \
    --exp_tag ${TAG} \
    --log_path logs/${MODEL}-${TAG}.log
}

# --- last_k 탐색 (lr 고정 1e-5) ---
run K1 1 1e-5
run K2 2 1e-5
run K3 3 1e-5

# --- best last_k에 대해 lr 탐색 ---
# K1~K3의 chrF를 보고 BEST를 best last_k 값으로 바꾼 뒤 아래 두 줄 실행.
BEST=2            # <-- 교체
run K4 ${BEST} 5e-6
run K5 ${BEST} 2e-5