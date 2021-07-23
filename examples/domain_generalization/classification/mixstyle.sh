#!/usr/bin/env bash
# PACS
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data/PACS -d PACS -s A C S -t P --mix-layers layer1 layer2 layer3 --freeze-bn --seed 0 --log logs/mixstyle/PACS_P
CUDA_VISIBLE_DEVICES=3 python mixstyle.py data/PACS -d PACS -s P C S -t A --mix-layers layer1 layer2 layer3 --freeze-bn --seed 0 --log logs/mixstyle/PACS_A
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data/PACS -d PACS -s P A S -t C --mix-layers layer1 layer2 layer3 --freeze-bn --seed 0 --log logs/mixstyle/PACS_C
CUDA_VISIBLE_DEVICES=1 python mixstyle.py data/PACS -d PACS -s P A C -t S --mix-layers layer1 layer2 layer3 --freeze-bn --seed 0 --log logs/mixstyle/PACS_S
# Office-Home
CUDA_VISIBLE_DEVICES=0 python mixstyle.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr --mix-layers layer1 layer2 --seed 0 --log logs/mixstyle/OfficeHome_Pr
CUDA_VISIBLE_DEVICES=1 python mixstyle.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw --mix-layers layer1 layer2 --seed 0 --log logs/mixstyle/OfficeHome_Rw
CUDA_VISIBLE_DEVICES=3 python mixstyle.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl --mix-layers layer1 layer2 --seed 0 --log logs/mixstyle/OfficeHome_Cl
CUDA_VISIBLE_DEVICES=4 python mixstyle.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar --mix-layers layer1 layer2 --seed 0 --log logs/mixstyle/OfficeHome_Ar
