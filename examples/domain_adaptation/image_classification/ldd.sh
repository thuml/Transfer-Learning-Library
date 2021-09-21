# ResNet50, Wilds Dataset
CUDA_VISIBLE_DEVICES=0 python ldd.py data/wilds -d fmow --train-resizing 'res.' --val-resizing 'res.' \
  -a densenet121 --bottleneck-dim 2048 --epochs 30 -i 1000 --seed 0 --log logs/ldd/fmow
