Domain-Adaptation-Lib

Training scripts for DANN on Office-31 UDA.
```bash
python examples/dann.py data/office31 -d Office31 -s A -t D -a resnet50 --gpu 0,1 -b 96 > result/dann_A2D.txt
python examples/dann.py data/office31 -d Office31 -s A -t W -a resnet50 --gpu 2,3 -b 96 > result/dann_A2W.txt
python examples/dann.py data/office31 -d Office31 -s W -t A -a resnet50 --gpu 2,3 -b 96 > result/dann_W2A.txt
python examples/dann.py data/office31 -d Office31 -s W -t D -a resnet50 --gpu 4,5 -b 96 > result/dann_W2D.txt
python examples/dann.py data/office31 -d Office31 -s D -t A -a resnet50 --gpu 4,5 -b 96 > result/dann_D2A.txt
python examples/dann.py data/office31 -d Office31 -s D -t W -a resnet50 --gpu 6,7 -b 96 > result/dann_D2W.txt

python examples/dann.py data/office31 -d Office31 -s A -t D -a resnet50 --gpu 6 -b 48 > result/dann_A2D_bs_48.txt
python examples/dann.py data/office31 -d Office31 -s A -t W -a resnet50 --gpu 6 -b 48 > result/dann_A2W_bs_48.txt
python examples/dann.py data/office31 -d Office31 -s W -t A -a resnet50 --gpu 6 -b 48 > result/dann_W2A_bs_48.txt
python examples/dann.py data/office31 -d Office31 -s W -t D -a resnet50 --gpu 7 -b 48 > result/dann_W2D_bs_48.txt
python examples/dann.py data/office31 -d Office31 -s D -t A -a resnet50 --gpu 7 -b 48 > result/dann_D2A_bs_48.txt
python examples/dann.py data/office31 -d Office31 -s D -t W -a resnet50 --gpu 7 -b 48 > result/dann_D2W_bs_48.txt
```

Training scripts for DANN on Office-Home UDA.
```bash
python examples/dann.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --gpu 0 -b 48 > result/dann_Ar2Cl.txt
python examples/dann.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --gpu 1 -b 48 > result/dann_Ar2Pr.txt
python examples/dann.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --gpu 1 -b 48 > result/dann_Ar2Rw.txt
python examples/dann.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --gpu 2 -b 48 > result/dann_Cl2Ar.txt
python examples/dann.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --gpu 2 -b 48 > result/dann_Cl2Pr.txt
python examples/dann.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --gpu 3 -b 48 > result/dann_Cl2Rw.txt
python examples/dann.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --gpu 3 -b 48 > result/dann_Pr2Ar.txt
python examples/dann.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --gpu 4 -b 48 > result/dann_Pr2Cl.txt
python examples/dann.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --gpu 4 -b 48 > result/dann_Pr2Rw.txt
python examples/dann.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --gpu 5 -b 48 > result/dann_Rw2Ar.txt
python examples/dann.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --gpu 5 -b 48 > result/dann_Rw2Cl.txt
python examples/dann.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --gpu 6 -b 48 > result/dann_Rw2Pr.txt
```

Training scripts for DANN on Office-Home UDA.
```bash
python examples/dann.py data/visda2017 -d VisDA2017 -s T -t V -a resnet101 --gpu 0,1,2,3 -b 144 > result/visda_resnet101_run1.txt
```


Training scripts for CDAN on Office-31 UDA.
```bash
python examples/cdan.py data/office31 -d Office31 -s A -t D -a resnet50 --gpu 0 -b 36 --i 150 --epochs 20 > result/cdan/office31_A2D.txt
python examples/cdan.py data/office31 -d Office31 -s A -t W -a resnet50 --gpu 1 -b 36 --i 150 --epochs 20 > result/cdan/office31_A2W.txt
python examples/cdan.py data/office31 -d Office31 -s W -t A -a resnet50 --gpu 2 -b 36 --i 150 --epochs 20 > result/cdan/office31_W2A.txt
python examples/cdan.py data/office31 -d Office31 -s W -t D -a resnet50 --gpu 3 -b 36 --i 150 --epochs 20 > result/cdan/office31_W2D.txt
python examples/cdan.py data/office31 -d Office31 -s D -t A -a resnet50 --gpu 4 -b 36 --i 150 --epochs 20 > result/cdan/office31_D2A.txt
python examples/cdan.py data/office31 -d Office31 -s D -t W -a resnet50 --gpu 5 -b 36 --i 150 --epochs 20 > result/cdan/office31_D2W.txt

python examples/cdan.py data/office31 -d Office31 -s A -t D -a resnet50 --gpu 6 -b 36 --i 150 --epochs 20 -E > result/cdan_e/office31_A2D.txt
python examples/cdan.py data/office31 -d Office31 -s A -t W -a resnet50 --gpu 6 -b 36 --i 150 --epochs 20 -E > result/cdan_e/office31_A2W.txt
python examples/cdan.py data/office31 -d Office31 -s W -t A -a resnet50 --gpu 6 -b 36 --i 150 --epochs 20 -E > result/cdan_e/office31_W2A.txt

python examples/cdan.py data/office31 -d Office31 -s W -t D -a resnet50 --gpu 7 -b 36 --i 150 --epochs 20 -E > result/cdan_e/office31_W2D.txt
python examples/cdan.py data/office31 -d Office31 -s D -t A -a resnet50 --gpu 7 -b 36 --i 150 --epochs 20 -E > result/cdan_e/office31_D2A.txt
python examples/cdan.py data/office31 -d Office31 -s D -t W -a resnet50 --gpu 7 -b 36 --i 150 --epochs 20 -E > result/cdan_e/office31_D2W.txt
```

Training scripts for CDAN on Office-Home UDA.
```bash
python examples/cdan.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --gpu 0 -b 36 -i 500 --epochs 20 > result/cdan/office_home_Ar2Cl.txt
python examples/cdan.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --gpu 0 -b 36 -i 500 --epochs 20 > result/cdan/office_home_Ar2Pr.txt
python examples/cdan.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --gpu 0 -b 36 -i 500 --epochs 20 > result/cdan/office_home_Ar2Rw.txt

python examples/cdan.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --gpu 1 -b 36 -i 500 --epochs 20 > result/cdan/office_home_Cl2Ar.txt
python examples/cdan.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --gpu 1 -b 36 -i 500 --epochs 20> result/cdan/office_home_Cl2Pr.txt
python examples/cdan.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --gpu 1 -b 36 -i 500 --epochs 20 > result/cdan/office_home_Cl2Rw.txt

python examples/cdan.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --gpu 2 -b 36 -i 500 --epochs 20 > result/cdan/office_home_Pr2Ar.txt
python examples/cdan.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --gpu 2 -b 36 -i 500 --epochs 20 > result/cdan/office_home_Pr2Cl.txt
python examples/cdan.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --gpu 2 -b 36 -i 500 --epochs 20 > result/cdan/office_home_Pr2Rw.txt

python examples/cdan.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --gpu 3 -b 36 -i 500 --epochs 20 > result/cdan/office_home_Rw2Ar.txt
python examples/cdan.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --gpu 3 -b 36 -i 500 --epochs 20 > result/cdan/office_home_Rw2Cl.txt
python examples/cdan.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --gpu 3 -b 36 -i 500 --epochs 20 > result/cdan/office_home_Rw2Pr.txt

python examples/cdan.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --gpu 6 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Ar2Cl.txt
python examples/cdan.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --gpu 6 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Ar2Pr.txt
python examples/cdan.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --gpu 6 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Ar2Rw.txt
python examples/cdan.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --gpu 6 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Cl2Ar.txt
python examples/cdan.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --gpu 6 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Cl2Pr.txt
python examples/cdan.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --gpu 6 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Cl2Rw.txt

python examples/cdan.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --gpu 7 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Pr2Ar.txt
python examples/cdan.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --gpu 7 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Pr2Cl.txt
python examples/cdan.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --gpu 7 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Pr2Rw.txt
python examples/cdan.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --gpu 7 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Rw2Ar.txt
python examples/cdan.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --gpu 7 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Rw2Cl.txt
python examples/cdan.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --gpu 7 -b 36 -i 500 --epochs 20 -E > result/cdan_e/office_home_Rw2Pr.txt
```

```bash
python examples/cdan.py data/visda -d VisDA2017 -s T -t V -a resnet50 --gpu 0,1 -b 120 -i 500 --epochs 60 > result/cdan/visda.txt
python examples/cdan.py data/visda -d VisDA2017 -s T -t V -a resnet50 --gpu 2,3 -b 120 -i 500 --epochs 60 -E > result/cdan_e/visda.txt

```


```bash
python examples/mdd.py data/office31 -d Office31 -s A -t W -a resnet50 --gpu 0 -b 36 --i 1000 --epochs 10  > result/mdd/office31_A2W.txt
python examples/mdd.py data/office31 -d Office31 -s D -t W -a resnet50 --gpu 1 -b 36 --i 1000 --epochs 10  > result/mdd/office31_D2W.txt
python examples/mdd.py data/office31 -d Office31 -s W -t D -a resnet50 --gpu 2 -b 36 --i 1000 --epochs 10  > result/mdd/office31_W2D.txt
python examples/mdd.py data/office31 -d Office31 -s A -t D -a resnet50 --gpu 3 -b 36 --i 1000 --epochs 10  > result/mdd/office31_A2D.txt
python examples/mdd.py data/office31 -d Office31 -s D -t A -a resnet50 --gpu 4 -b 36 --i 1000 --epochs 10  > result/mdd/office31_D2A.txt
python examples/mdd.py data/office31 -d Office31 -s W -t A -a resnet50 --gpu 5 -b 36 --i 1000 --epochs 10  > result/mdd/office31_W2A.txt


python examples/mdd.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --gpu 0 -b 36 --i 1000 --epochs 20  > result/mdd/office_home_Ar2Cl.txt

```

