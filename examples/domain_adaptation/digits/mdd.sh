CUDA_VISIBLE_DEVICES=1 python mdd.py data/mnist data/usps -s MNIST -t USPS -a lenet --seed 0 --log logs/mdd/MNIST2USPS
CUDA_VISIBLE_DEVICES=1 python mdd.py data/usps data/mnist -s USPS -t MNIST -a lenet --seed 0 --log logs/mdd/USPS2MNIST
CUDA_VISIBLE_DEVICES=1 python mdd.py data/svhn data/mnist -s SVHN -t MNIST -a dtn --image-size 32 --num-channels 3 --seed 0 --log logs/mdd/SVHN2MNIST
