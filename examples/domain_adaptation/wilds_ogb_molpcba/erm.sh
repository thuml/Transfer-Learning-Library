# ogb-molpcba
CUDA_VISIBLE_DEVICES=0 python erm.py data/wilds --lr 3e-2 -b 4096 4096 --epochs 200 \
  --seed 0 --deterministic --log logs/erm/obg_lr_0_03_deterministic
