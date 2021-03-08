# GTA5 to Cityscapes
# First, train the CycleGAN
CUDA_VISIBLE_DEVICES=0 python cycada.py data/GTA5 data/Cityscapes -s GTA5 -t Cityscapes \
    --log logs/cycada/gtav2cityscapes --pretrain logs/src_only/gtav2cityscapes/checkpoints/59.pth \
    --translated-root data/GTA52Cityscapes/cycada_39
# Then, train the src_only model on the translated source dataset
CUDA_VISIBLE_DEVICES=0 python source_only.py data/GTA52Cityscapes/cycada_39 data/Cityscapes \
    -s GTA5 -t Cityscapes --log logs/cycada_src_only/gtav2cityscapes


## Synthia to Cityscapes
# First, train the Cycada
CUDA_VISIBLE_DEVICES=0 python cycada.py data/synthia data/Cityscapes -s Synthia -t Cityscapes \
    --log logs/cycada/synthia2cityscapes --pretrain logs/src_only/synthia2cityscapes/checkpoints/59.pth \
    --translated-root data/Synthia2Cityscapes/cycada_39
# Then, train the src_only model on the translated source dataset
CUDA_VISIBLE_DEVICES=0 python source_only.py data/Synthia2Cityscapes/cycada_39 data/Cityscapes \
    -s Synthia -t Cityscapes --log logs/cycada_src_only/synthia2cityscapes


# Cityscapes to FoggyCityscapes
# First, train the CycleGAN
CUDA_VISIBLE_DEVICES=0 python cycada.py data/Cityscapes data/Cityscapes -s Cityscapes -t FoggyCityscapes \
    --log logs/cycada/cityscapes2foggy --pretrain logs/src_only/cityscapes2foggy/checkpoints/59.pth \
    --translated-root data/Cityscapes2Foggy/cycada_39
# Then, train the src_only model on the translated source dataset
CUDA_VISIBLE_DEVICES=0 python source_only.py data/Cityscapes2Foggy/cycada_39 data/Cityscapes \
    -s Cityscapes -t FoggyCityscapes --log logs/cycada_src_only/cityscapes2foggy
