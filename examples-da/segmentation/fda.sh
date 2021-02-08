CUDA_VISIBLE_DEVICES=0 python fda.py data/GTA5 data/Cityscapes -s GTA5 -t Cityscapes \
    --log logs/fda/gtav2cityscapes --debug
CUDA_VISIBLE_DEVICES=0 python fda.py data/synthia data/Cityscapes -s Synthia -t Cityscapes \
    --log logs/fda/synthia2cityscapes --debug
