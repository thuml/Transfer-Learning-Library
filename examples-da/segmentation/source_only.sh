CUDA_VISIBLE_DEVICES=0 python source_only.py data/GTA5 data/Cityscapes \
    -s GTA5 -t Cityscapes --log logs/src_only/gtav2cityscapes
CUDA_VISIBLE_DEVICES=0 python source_only.py data/synthia data/Cityscapes \
    -s Synthia -t Cityscapes --log logs/src_only/synthia2cityscapes


CUDA_VISIBLE_DEVICES=0 python source_only.py data/Cityscapes data/Cityscapes \
    -s Cityscapes -t Cityscapes --log logs/oracle/cityscapes

