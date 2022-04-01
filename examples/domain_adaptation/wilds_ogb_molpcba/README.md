# Unsupervised Domain Adapatation for WILDS (Molecule classification)

## Installation
It's suggeste to use **pytorch==1.10.1** in order to reproduce the benchmark results.

You need to run
```
pip install -r requirements.txt
```

## Dataset

Following datasets can be downloaded automatically:
- [OGB-MolPCBA (WILDS)](https://wilds.stanford.edu/datasets/)

## Supported Methods

TODO

## Usage
The shell files give all the training scripts we use, e.g.
```
CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds --arch 'gin_virtual' \
    --deterministic --log logs/erm/obg --lr 3e-2 --wd 0.0 \
    --epochs 200 --metric ap -b 4096 4096 --seed 0
```

## Results

### Performance on WILDS-OGB-MolPCBA (GIN-virtual)
| Methods | Val Avg Precision | Test Avg Precision | GPU Memory Usage(GB)|
| --- | --- | --- | --- |
| ERM | 29.0 | 28.0 | 17.8 |

### Visuialization
We use tensorboard to record the training process and visualize the outputs of the models. 
```
tensorboard --logdir=logs
```
<img src="./fig/ogb-molpcba_train_loss.png" width="300"/>
