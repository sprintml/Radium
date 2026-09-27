#!/bin/bash

eval "$(conda shell.bash hook)"
source ~/.bashrc

architecture=infinity # infinity, instella_iar
num_samples=5000


#clean_dir="/path/datasets/mscoco2014val/val2014/"
case "$architecture" in
    "infinity")
        conda activate inf_water
        ;;
    "instella_iar")
        conda activate instella_new
        ;;
    "big_r")
        conda activate BiGR
        ;;
    *)
        echo "Error: Unsupported architecture: $architecture"
        echo "Supported architectures: infinity, instella_iar, big_r"
        exit 1
        ;;
esac
# debug print-outs
echo USER: $USER
which conda
which python

# INSTELLA IAR ARGUMENTS
config='./Instella-T2I/configs/ar_config.py'
ckpt_path='./Instella-T2I/checkpoints'

image_size=512 # 256
if [ $image_size -eq 256 ]; then
    model='BiGR-L-d24' # Model name
    seq_len=256 # 256
    ckpt_bae='a'
    ckpt='a'
else
    seq_len=1024 # 256
    ckpt_bae='/path/BiGR/pretrained_models/binaryae_ema_720000.th' # Path to the BAE checkpoint
    model='BiGR-L-512' # Model name
    ckpt='/path/BiGR/pretrained_models/bigr_L_d32.pt' # Path to the model checkpoint

fi
num_sample_iter=10 # Unmasking steps
sampling_steps=10 # t
gumbel_temp=0.01 # Temperature for Gumbel sampling
cfg=2.5 # Configuration for the model

if [ "$architecture" == "infinity" ]; then
    pn=1M
    model_type=infinity_2b
    use_scale_schedule_embedding=0
    use_bit_label=1
    checkpoint_type='torch'
    infinity_model_path=/path/infinity/infinity_2b_reg.pth #8.4GB
    vae_type=32
    vae_path=/path/infinity/infinity_vae_d32reg.pth
    cfg=3
    tau=0.5
    rope2d_normalized_by_hw=2
    add_lvl_embeding_only_first_block=1
    rope2d_each_sa_layer=1
    text_encoder_ckpt=/path/models/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001
    text_channels=2048
    apply_spatial_patchify=0
fi
watermark_scales=2 #0: No watermark, 1: Only last scale, 2: Apply on all scales, 3: up to the 9th scale, 4: from the 10th scale up, 5: [3,4,5]
watermark_delta=2 #Bias model towards green set
watermark_context_width=2 # When changed, make sure to check if it actually still works properly. This hasnt been touched in a while
watermark_count_bit_loss_after_reencoding=0 # reencodes the generated image and counts the bits overlap. Can also count the token overlap, but the flag is hardcoded. Adds inference speed because the generated image has to be reencoded, so only use if needed
watermark_method='2-bit_pattern' # 2-bit_pattern or N-bit_pattern
clean_dir="/path/${architecture}/${model}/delta_0"

stable_diff_vae="/path/models/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493"
# run the code
set="01,10"
watermarked_dir="/path/${architecture}/${model}/delta_${1}"
clean_dir="/path/${architecture}/${model}/delta_0"

case "$architecture" in
    "infinity")
        echo "Running Infinity architecture"
        PYTHONPATH=. python3 robustness_test.py \
        --cfg ${cfg} \
        --tau ${tau} \
        --pn ${pn} \
        --model_path ${infinity_model_path} \
        --vae_type ${vae_type} \
        --vae_path ${vae_path} \
        --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
        --use_bit_label ${use_bit_label} \
        --model_type ${model_type} \
        --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
        --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
        --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
        --cfg ${cfg} \
        --tau ${tau} \
        --checkpoint_type ${checkpoint_type} \
        --text_channels ${text_channels} \
        --apply_spatial_patchify ${apply_spatial_patchify} \
        --seed 0 \
        --watermark_scales ${watermark_scales} \
        --watermark_delta ${watermark_delta} \
        --watermark_context_width ${watermark_context_width} \
        --watermark_count_bit_loss_after_reencoding ${watermark_count_bit_loss_after_reencoding} \
        --watermark_method ${watermark_method} \
        --watermarked_dir ${watermarked_dir} \
        --clean_dir ${clean_dir} \
        --stable_diff_vae ${stable_diff_vae} \
        --num_samples ${num_samples} \
        --model_folder_path /path/models \
        --set "${set}"

    ;;
    "instella_iar")
        echo "Running instella IAR architecture"
        PYTHONPATH=. python3 robustness_test.py \
        --seed 0 \
        --architecture ${architecture} \
        --clean_dir ${clean_dir} \
        --stable_diff_vae ${stable_diff_vae} \
        --num_samples ${num_samples} \
        --watermark_context_width ${watermark_context_width} \
        --watermark_method ${watermark_method} \
        --watermarked_dir ${watermarked_dir} \
        --model_folder_path /path/models \
        --config ${config} \
        --ckpt_path ${ckpt_path} \
        --set "${set}" 
    ;;
    "big_r")
        echo "Running BiGR architecture"
        PYTHONPATH=. python3 robustness_test.py \
        --seed 0 \
        --architecture ${architecture} \
        --clean_dir ${clean_dir} \
        --stable_diff_vae ${stable_diff_vae} \
        --num_samples ${num_samples} \
        --watermark_context_width ${watermark_context_width} \
        --watermark_method ${watermark_method} \
        --watermarked_dir ${watermarked_dir} \
        --model_folder_path /path/models \
        --set "${set}" \
        --image_size ${image_size} \
        --img_size ${image_size} \
        --seq_len ${seq_len} \
        --ckpt_bae ${ckpt_bae} \
        --model ${model} \
        --ckpt ${ckpt} \
        --num_sample_iter ${num_sample_iter} \
        --gumbel_temp ${gumbel_temp} \
        --cfg ${cfg} \
        --infer_steps ${sampling_steps} \
        ;;
    *)
        echo "Error: Unsupported architecture: $architecture"
        echo "Supported architectures: infinity, instella_iar"
        exit 1
        ;;
esac
