CUDA_VISIBLE_DEVICES=2 python dwa.py data/domainnet -d DomainNetv2 -tr c i p q r s -ts c i p q r s -a resnet101 \
  --epochs 20 --seed 0  --log logs/dwa/DomainNet
