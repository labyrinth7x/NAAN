GPUS=2
export CUDA_VISIBLE_DEVICES=$GPUS
python3.5 eval_gat.py \
            --k_hops 20 10 \
            --sample_hops 20 \
            --nheads 1 \
            --split split1 \
            --step 0.6 \
            --th 0.8 \
            --logs_dir results/lr-0.002-wd-5e-4-hops-100-10-batch-512-nheads-1-alpha-0.2-dropout-0-clip-2-120-0 \
            --num_workers 4 \
            --model_path epoch_2_step_50000.ckpt \
