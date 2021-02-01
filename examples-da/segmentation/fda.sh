CUDA_VISIBLE_DEVICES=0 python examples-da/segmentation/fda.py data/GTA5 data/Cityscapes -s GTA5 -t Cityscapes --snapshot-dir snapshots/da/segmentation/fda/gtav2cityscapes
CUDA_VISIBLE_DEVICES=0 python examples-da/segmentation/fda.py data/GTA5 data/Cityscapes -s GTA5 -t Cityscapes --entropy-weight 0.005 --snapshot-dir snapshots/da/segmentation/fda/gtav2cityscapes_entropy_0_005

CUDA_VISIBLE_DEVICES=0 python examples-da/segmentation/fda.py data/synthia data/Cityscapes -s Synthia -t Cityscapes --snapshot-dir snapshots/da/segmentation/fda/synthia2cityscapes
CUDA_VISIBLE_DEVICES=0 python examples-da/segmentation/fda.py data/synthia data/Cityscapes -s Synthia -t Cityscapes --entropy-weight 0.005 --snapshot-dir snapshots/da/segmentation/fda/synthia2cityscapes_entropy_0_005