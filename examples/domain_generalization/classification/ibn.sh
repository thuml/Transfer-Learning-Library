#!/usr/bin/env bash
# PACS
CUDA_VISIBLE_DEVICES=4 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet50_ibn_b --freeze-bn --seed 0 --log logs/baseline/PACS_P
CUDA_VISIBLE_DEVICES=5 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet50_ibn_b --freeze-bn --seed 0 --log logs/baseline/PACS_A
CUDA_VISIBLE_DEVICES=6 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet50_ibn_b --freeze-bn --seed 0 --log logs/baseline/PACS_C
CUDA_VISIBLE_DEVICES=7 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50_ibn_b --freeze-bn --seed 0 --log logs/baseline/PACS_S
# Office-Home
CUDA_VISIBLE_DEVICES=4 python baseline.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50_ibn_b --seed 0 --log logs/baseline/OfficeHome_Pr
CUDA_VISIBLE_DEVICES=5 python baseline.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50_ibn_b --seed 0 --log logs/baseline/OfficeHome_Rw
CUDA_VISIBLE_DEVICES=6 python baseline.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl -a resnet50_ibn_b --seed 0 --log logs/baseline/OfficeHome_Cl
CUDA_VISIBLE_DEVICES=7 python baseline.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar -a resnet50_ibn_b --seed 0 --log logs/baseline/OfficeHome_Ar