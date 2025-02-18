#!/bin/bash

# Define lists of values
datasets=(imagenet)
shots=(1 4 16)
seeds=(1 42 100)
rhos=(0.05 0.1 0.2 0.4 0.75 1.5)
gpu_id=3

# Loop over combinations of values
for dataset in "${datasets[@]}"; do
    for shot in "${shots[@]}"; do
        for seed in "${seeds[@]}"; do
            for rho in "${rhos[@]}"; do
                rho_name=$(echo "$rho * 100" | bc | awk '{printf "%d", $0}')
                filename="lora_weights_adaptive_${rho_name}"
                echo "$filename"
                echo "Running with dataset=$dataset, shot=$shot, seed=$seed, rho=$rho, filename=$filename"
                CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                    --root_path /home/scratch/arshv/CLIP-LoRA/data \
                    --dataset "$dataset" \
                    --shots "$shot" \
                    --seed "$seed" \
                    --save_path checkpoints \
                    --do_SAM \
                    --rho "$rho" \
                    --filename "$filename" \
                    --adaptive \
                    --logfile "logs/${dataset}_results_adaptive.log"
            done
        done
    done
done
