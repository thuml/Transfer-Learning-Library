# ResNet50, Wilds Dataset
CUDA_VISIBLE_DEVICES=0 python ldd.py data/wilds -d fmow --train-resizing 'res.' --val-resizing 'res.' \
  -a densenet121 --bottleneck-dim 2048 --epochs 30 -i 1000 --seed 0 --log logs/ldd/fmow

# Digits
CUDA_VISIBLE_DEVICES=0 python ldd.py data/digits -d Digits -s MNIST -t USPS --train-resizing 'res.' --val-resizing 'res.' \
  --resize-size 28 --no-hflip --norm-mean 0.5 --norm-std 0.5 -a lenet --no-pool -b 128 -i 2500 --scratch --seed 0 --log logs/ldd/MNIST2USPS
CUDA_VISIBLE_DEVICES=0 python ldd.py data/digits -d Digits -s USPS -t MNIST --train-resizing 'res.' --val-resizing 'res.' \
  --resize-size 28 --no-hflip --norm-mean 0.5 --norm-std 0.5 -a lenet --no-pool -b 128 -i 2500 --scratch --seed 0 --log logs/ldd/USPS2MNIST
CUDA_VISIBLE_DEVICES=0 python ldd.py data/digits -d Digits -s SVHNRGB -t MNISTRGB --train-resizing 'res.' --val-resizing 'res.' \
  --resize-size 32 --no-hflip --norm-mean 0.5 0.5 0.5 --norm-std 0.5 0.5 0.5 -a dtn --no-pool -b 128 -i 2500 --scratch --seed 0 --log logs/ldd/SVHN2MNIST
