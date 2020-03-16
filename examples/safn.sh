#!/usr/bin/env bash
# Office31
python examples/safn.py data/office31 -d Office31 -s A -t W -a resnet50 --gpu 0 --epochs 20  > benchmarks/safn/Office31_A2W.txt
python examples/safn.py data/office31 -d Office31 -s D -t W -a resnet50 --gpu 0 --epochs 20  > benchmarks/safn/Office31_D2W.txt
python examples/safn.py data/office31 -d Office31 -s W -t D -a resnet50 --gpu 0 --epochs 20  > benchmarks/safn/Office31_W2D.txt
python examples/safn.py data/office31 -d Office31 -s A -t D -a resnet50 --gpu 0 --epochs 20  > benchmarks/safn/Office31_A2D.txt
python examples/safn.py data/office31 -d Office31 -s D -t A -a resnet50 --gpu 0 --epochs 20  > benchmarks/safn/Office31_D2A.txt
python examples/safn.py data/office31 -d Office31 -s W -t A -a resnet50 --gpu 0 --epochs 20  > benchmarks/safn/Office31_W2A.txt

# Office-Home
python examples/safn.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --gpu 1 --epochs 40 > benchmarks/safn/OfficeHome_Ar2Cl.txt
python examples/safn.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --gpu 1 --epochs 40 > benchmarks/safn/OfficeHome_Ar2Pr.txt
python examples/safn.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --gpu 2 --epochs 40 > benchmarks/safn/OfficeHome_Ar2Rw.txt
python examples/safn.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --gpu 2 --epochs 40 > benchmarks/safn/OfficeHome_Cl2Ar.txt
python examples/safn.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --gpu 3 --epochs 40 > benchmarks/safn/OfficeHome_Cl2Pr.txt
python examples/safn.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --gpu 3 --epochs 40 > benchmarks/safn/OfficeHome_Cl2Rw.txt
python examples/safn.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --gpu 4 --epochs 40 > benchmarks/safn/OfficeHome_Pr2Ar.txt
python examples/safn.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --gpu 4 --epochs 40 > benchmarks/safn/OfficeHome_Pr2Cl.txt
python examples/safn.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --gpu 5 --epochs 40 > benchmarks/safn/OfficeHome_Pr2Rw.txt
python examples/safn.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --gpu 5 --epochs 40 > benchmarks/safn/OfficeHome_Rw2Ar.txt
python examples/safn.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --gpu 6 --epochs 40 > benchmarks/safn/OfficeHome_Rw2Cl.txt
python examples/safn.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --gpu 6 --epochs 40 > benchmarks/safn/OfficeHome_Rw2Pr.txt

# VisDA-2017
python examples/safn.py data/visda-2017 -d VisDA2017 -s T -t V -a resnet50 --gpu 2 --epochs 50 > benchmarks/safn/VisDA2017.txt
