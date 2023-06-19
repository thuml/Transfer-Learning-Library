
# PCGrad
CUDA_VISIBLE_DEVICES=0 python pcgrad.py data/domainnet -d DomainNetv2 -s c i p q r s -t c i p q r s -a resnet101 \
  --epochs 20 --seed 0  --log logs/PCGrad/DomainNet

CUDA_VISIBLE_DEVICES=0 python pcgrad.py data/domainnet -d DomainNetv2 -s c i p q r s -t c i p q r s -a resnet101 \
  --epochs 20 --seed 0  --log logs/PCGrad_3x_lr/DomainNet --lr 0.03

CUDA_VISIBLE_DEVICES=0 python pcgrad.py data/domainnet -d DomainNetv2 -s c i p q r s -t c i p q r s -a resnet101 \
  --epochs 20 --seed 0  --log logs/PCGrad_bs_48/DomainNet -b 48
