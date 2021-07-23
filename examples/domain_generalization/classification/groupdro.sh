#!/usr/bin/env bash
# PACS
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/PACS -d PACS -s A C S -t P --freeze-bn --seed 0 --log logs/groupdro/PACS_P
CUDA_VISIBLE_DEVICES=1 python groupdro.py data/PACS -d PACS -s P C S -t A --freeze-bn --seed 0 --log logs/groupdro/PACS_A
CUDA_VISIBLE_DEVICES=3 python groupdro.py data/PACS -d PACS -s P A S -t C --freeze-bn --seed 0 --log logs/groupdro/PACS_C
CUDA_VISIBLE_DEVICES=4 python groupdro.py data/PACS -d PACS -s P A C -t S --freeze-bn --seed 0 --log logs/groupdro/PACS_S
# Office-Home
CUDA_VISIBLE_DEVICES=0 python groupdro.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr --seed 0 --log logs/groupdro/OfficeHome_Pr
CUDA_VISIBLE_DEVICES=1 python groupdro.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw --seed 0 --log logs/groupdro/OfficeHome_Rw
CUDA_VISIBLE_DEVICES=3 python groupdro.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl --seed 0 --log logs/groupdro/OfficeHome_Cl
CUDA_VISIBLE_DEVICES=4 python groupdro.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar --seed 0 --log logs/groupdro/OfficeHome_Ar