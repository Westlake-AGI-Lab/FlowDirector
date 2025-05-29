torchrun --nproc_per_node=4 \
    generate.py \
    --task t2v-1.3B \
    --size 832*480 \
    --ckpt_dir /root/autodl-tmp/Wan2.1-T2V-1.3B \
    --base_seed 1 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 4 \
    --sample_solver "unipc" \
    --prompt "A cute and adorable fluffy puppy wearing a witch hat in a halloween autumn evening forest." \
    --sample_guide_scale 6 \
    --sample_shift 12   