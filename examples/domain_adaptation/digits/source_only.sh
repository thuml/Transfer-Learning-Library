CUDA_VISIBLE_DEVICES=1 python source_only.py data/mnist data/usps -s MNIST -t USPS -a lenet --seed 0 --log logs/src_only/MNIST2USPS
CUDA_VISIBLE_DEVICES=1 python source_only.py data/usps data/mnist -s USPS -t MNIST -a lenet --seed 0 --log logs/src_only/USPS2MNIST
CUDA_VISIBLE_DEVICES=1 python source_only.py data/svhn data/mnist -s SVHN -t MNIST -a dtn --image-size 32 --num-channels 3 --seed 0 --log logs/src_only/SVHN2MNIST
