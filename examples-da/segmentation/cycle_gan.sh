# First, train the CycleGAN
CUDA_VISIBLE_DEVICES=7 python cycle_gan.py data/GTA5 data/Cityscapes -s GTA5 -t Cityscapes \
    --log logs/cyclegan/gtav2cityscapes
# Translate the source dataset into the target style using the pretrained CycleGAN
CUDA_VISIBLE_DEVICES=0 python cycle_gan.py data/GTA5 data/Cityscapes -s GTA5 -t Cityscapes \
    --log logs/cyclegan/gtav2cityscapes --resume logs/cycle_gan/gtav2cityscapes/*.pth \
    --test-only --translated-root data/GTA52Cityscapes/CycleGAN_*
# Finally, train the advent model on the translated source dataset and the original target dataset
CUDA_VISIBLE_DEVICES=0 python advent.py data/GTA5 data/GTA52Cityscapes/CycleGAN_* \
    -s GTA5 -t Cityscapes --log logs/cyclegan_advent/gtav2cityscapes




