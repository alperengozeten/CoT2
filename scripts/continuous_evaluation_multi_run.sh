#!/usr/bin/env bash
digit_range_1=1
digit_range_2=10
batch_size=16
num_layers=2
num_heads=2
seq_length=4
max_seq_len=$(( 2 * seq_length + 3 ))  # extra <BOS>, ->, <EOS> tokens
split_method=random_permutation
num_epochs=1000
embedding_dims=(16 24 32)
supervisions=(hard_teacher)
curriculum_epoch=150
loss_steps=(0 1 2 h)
dist_strategy=uniform  # uniform (equivalent to num_beams=all) or beam_minabs
num_beams=16
seeds=(42 43 44 45 46)
gpus=(0 1 2 3 4) 
output_dir=continuous_model_multi_run_digit_range_${digit_range_1}_${digit_range_2}_Seq_Length_${seq_length}_Layer_${num_layers}_Head_${num_heads}_Dist_Strategy_${dist_strategy}_Num_Beams_${num_beams}/

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
    for supervision in "${supervisions[@]}"
    do
        for i in "${!seeds[@]}"
        do
            seed=${seeds[$i]}
            gpu=${gpus[$i]}
            nohup_out=logs/${output_dir}Embed${embedding_dim}_${split_method}_${supervision}_${batch_size}_Seed${seed}.out

            CUDA_VISIBLE_DEVICES=$gpu \
            nohup python -u "$REPO_ROOT/train/continuous_train.py" \
            --max_seq_len $max_seq_len \
            --embedding_dim $embedding_dim \
            --digit_range $digit_range_1 $digit_range_2 \
            --batch_size $batch_size \
            --seq_length $seq_length \
            --loss_steps "${loss_steps[@]}" \
            --split_method $split_method \
            --num_epochs $num_epochs \
            --output_dir $output_dir \
            --supervision $supervision \
            --curriculum_epoch $curriculum_epoch \
            --num_layers $num_layers \
            --num_heads $num_heads \
            --seed $seed \
            > "$nohup_out" 2>&1 &

            echo "nohup output: $nohup_out"
        done
    done
done