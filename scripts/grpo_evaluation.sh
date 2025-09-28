#!/usr/bin/env bash
digit_range_1=1
digit_range_2=10
batch_size=16
num_layers=1
num_heads=1
seq_length=4
max_seq_len=$(( 2 * seq_length + 3 ))  # extra <BOS>, ->, <EOS> tokens
split_method=random_permutation
num_epochs=100
embedding_dims=(32)
num_rollouts=8
K_values=(2 3 6)
kl_coeff=0.0
clip_epsilon=0.1
output_dir=grpo_model_digit_range_${digit_range_1}_${digit_range_2}_Seq_Length_${seq_length}_Layer_${num_layers}_Head_${num_heads}/
sampling_normalization=true

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

[ ! -d "logs" ] && mkdir logs
[ ! -d "models" ] && mkdir models
[ ! -d "figures" ] && mkdir figures

if [ ! -d "figures/${output_dir}" ]; then
    mkdir figures/${output_dir}
fi
if [ ! -d "logs/${output_dir}" ]; then
    mkdir logs/${output_dir}
fi
if [ ! -d "models/${output_dir}" ]; then
    mkdir models/${output_dir}
fi


for embedding_dim in "${embedding_dims[@]}"
do
    for K in "${K_values[@]}"
    do
        nohup_out=logs/${output_dir}Embed${embedding_dim}_K${K}_${split_method}_${batch_size}.out

        CUDA_VISIBLE_DEVICES=0 \
        nohup python -u "$REPO_ROOT/train/continuous_train_grpo_mts.py" \
        --max_seq_len $max_seq_len \
        --embedding_dim $embedding_dim \
        --digit_range $digit_range_1 $digit_range_2 \
        --batch_size $batch_size \
        --seq_length $seq_length \
        --split_method $split_method \
        --num_epochs $num_epochs \
        --output_dir $output_dir \
        --num_layers $num_layers \
        --num_heads $num_heads \
        --num_rollouts $num_rollouts \
        --K $K \
        --kl_coeff $kl_coeff \
        --clip_epsilon $clip_epsilon \
        --sampling_normalization $sampling_normalization \
        > "$nohup_out" 2>&1 &

        echo "nohup output: $nohup_out"
    done
done