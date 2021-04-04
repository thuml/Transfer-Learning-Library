# GTA5 to Cityscapes
# First, train the CycleGAN
CUDA_VISIBLE_DEVICES=0 python cycle_gan.py data/GTA5 data/Cityscapes -s GTA5 -t Cityscapes \
    --log logs/cyclegan/gtav2cityscapes --translated-root data/GTA52Cityscapes/CycleGAN_39
# Then, train the src_only model on the translated source dataset
CUDA_VISIBLE_DEVICES=0 python source_only.py data/GTA52Cityscapes/CycleGAN_39 data/Cityscapes \
    -s GTA5 -t Cityscapes --log logs/cyclegan_src_only/gtav2cityscapes


# Cityscapes to FoggyCityscapes
# First, train the CycleGAN
CUDA_VISIBLE_DEVICES=0 python cycle_gan.py data/Cityscapes data/Cityscapes -s Cityscapes -t FoggyCityscapes \
    --log logs/cyclegan/cityscapes2foggy --translated-root data/Cityscapes2Foggy/CycleGAN_39
# Then, train the src_only model on the translated source dataset
CUDA_VISIBLE_DEVICES=0 python source_only.py data/Cityscapes2Foggy/CycleGAN_39 data/Cityscapes \
    -s Cityscapes -t FoggyCityscapes --log logs/cyclegan_src_only/cityscapes2foggy
