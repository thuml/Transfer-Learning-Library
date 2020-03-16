#!/usr/bin/env bash
# Office31
python examples/mcd.py data/office31 -d Office31 -s A -t W -a resnet50 --gpu 0 --epochs 10 > benchmarks/mcd/Office31_A2W.txt
python examples/mcd.py data/office31 -d Office31 -s D -t W -a resnet50 --gpu 0 --epochs 10 > benchmarks/mcd/Office31_D2W.txt
python examples/mcd.py data/office31 -d Office31 -s W -t D -a resnet50 --gpu 0 --epochs 10 > benchmarks/mcd/Office31_W2D.txt
python examples/mcd.py data/office31 -d Office31 -s A -t D -a resnet50 --gpu 0 --epochs 10 > benchmarks/mcd/Office31_A2D.txt
python examples/mcd.py data/office31 -d Office31 -s D -t A -a resnet50 --gpu 0 --epochs 10 > benchmarks/mcd/Office31_D2A.txt
python examples/mcd.py data/office31 -d Office31 -s W -t A -a resnet50 --gpu 0 --epochs 10 > benchmarks/mcd/Office31_W2A.txt

# Office-Home
python examples/mcd.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --gpu 1 --epochs 20 > benchmarks/mcd/OfficeHome_Ar2Cl.txt
python examples/mcd.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --gpu 1 --epochs 20 > benchmarks/mcd/OfficeHome_Ar2Pr.txt
python examples/mcd.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --gpu 1 --epochs 20 > benchmarks/mcd/OfficeHome_Ar2Rw.txt
python examples/mcd.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --gpu 2 --epochs 20 > benchmarks/mcd/OfficeHome_Cl2Ar.txt
python examples/mcd.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --gpu 2 --epochs 20 > benchmarks/mcd/OfficeHome_Cl2Pr.txt
python examples/mcd.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --gpu 2 --epochs 20 > benchmarks/mcd/OfficeHome_Cl2Rw.txt
python examples/mcd.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --gpu 3 --epochs 20 > benchmarks/mcd/OfficeHome_Pr2Ar.txt
python examples/mcd.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --gpu 3 --epochs 20 > benchmarks/mcd/OfficeHome_Pr2Cl.txt
python examples/mcd.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --gpu 3 --epochs 20 > benchmarks/mcd/OfficeHome_Pr2Rw.txt
python examples/mcd.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --gpu 4 --epochs 20 > benchmarks/mcd/OfficeHome_Rw2Ar.txt
python examples/mcd.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --gpu 4 --epochs 20 > benchmarks/mcd/OfficeHome_Rw2Cl.txt
python examples/mcd.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --gpu 4 --epochs 20 > benchmarks/mcd/OfficeHome_Rw2Pr.txt

# VisDA2017
python examples/mcd.py data/visda -d VisDA2017 -s T -t V -a resnet50 --gpu 7 --epochs 40 > benchmarks/mcd/VisDA2017.txt
python examples/mcd.py data/visda -d VisDA2017 -s T -t V -a resnet101 --gpu 5,6 --epochs 40 > benchmarks/mcd/VisDA2017_resnet101.txt


