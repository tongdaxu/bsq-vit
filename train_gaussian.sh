export WANDB_API_KEY=589d531f376ac4cfc8d713753b96052a1420b709
export http_proxy=httpproxy.glm.ai:8888
export https_proxy=httpproxy.glm.ai:8888
export TORCH_HOME=/workspace/cogview_dev/xutd/xu/models

python -m torch.distributed.launch --standalone \
    --nproc-per-node=8 main_image_tokenizer.py \
    --config=/workspace/cogview_dev/xutd/xu/bsq-vit/configs/tokenizer/imagenet_256x256_ta_t_16_g_16_stylegan_f8_fp16_lr_2e-7.yaml \
    --output-dir=./logs/imagenet_256x256_ta_t_16_g_16_stylegan_f8_fp16_lr_2e-7
