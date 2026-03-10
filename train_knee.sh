mkdir -p logs

nohup python /home/chenfurong/CODE2025/Mamba2025/train_knee.py \
    --root_path /public/home/chenfurong/2025data/fastmri_knee/ \
    --max_iterations 100000 \
    --gpu 1 \
    --model_name dmsmamba \
    --batch_size 4 \
    --base_lr 0.001 \
    --MRIDOWN 8X \
    --low_field_SNR 0 \
    --kspace_refine False \
    --use_multi_modal True \
    --modality t2 \
    --input_normalize mean_std\
    --exp DMSMamba_lr001_8X_knee \
    > logs/$(date +"%Y%m%d-%H%M")_DMSMamba_lr001_8X_knee.log 2>&1 &