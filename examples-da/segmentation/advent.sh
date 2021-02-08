CUDA_VISIBLE_DEVICES=0 python advent.py data/GTA5 data/Cityscapes -s GTA5 -t Cityscapes \
    --log logs/advent/gtav2cityscapes
CUDA_VISIBLE_DEVICES=0 python advent.py data/synthia data/Cityscapes -s Synthia -t Cityscapes \
    --log logs/advent/synthia2cityscapes