# DSprites
CUDA_VISIBLE_DEVICES=4 python mdd.py data/dSprites -d DSprites -s C -t N -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_C2N --wd 0.0005
CUDA_VISIBLE_DEVICES=4 python mdd.py data/dSprites -d DSprites -s C -t S -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_C2S --wd 0.0005
CUDA_VISIBLE_DEVICES=4 python mdd.py data/dSprites -d DSprites -s N -t C -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_N2C --wd 0.0005
CUDA_VISIBLE_DEVICES=4 python mdd.py data/dSprites -d DSprites -s N -t S -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_N2S --wd 0.0005
CUDA_VISIBLE_DEVICES=4 python mdd.py data/dSprites -d DSprites -s S -t C -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_S2C --wd 0.0005
CUDA_VISIBLE_DEVICES=4 python mdd.py data/dSprites -d DSprites -s S -t N -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_S2N --wd 0.0005

CUDA_VISIBLE_DEVICES=3 python mdd.py data/dSprites -d DSprites -s C -t N -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_C2N
CUDA_VISIBLE_DEVICES=3 python mdd.py data/dSprites -d DSprites -s C -t S -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_C2S
CUDA_VISIBLE_DEVICES=3 python mdd.py data/dSprites -d DSprites -s N -t C -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_N2C
CUDA_VISIBLE_DEVICES=3 python mdd.py data/dSprites -d DSprites -s N -t S -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_N2S
CUDA_VISIBLE_DEVICES=3 python mdd.py data/dSprites -d DSprites -s S -t C -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_S2C
CUDA_VISIBLE_DEVICES=3 python mdd.py data/dSprites -d DSprites -s S -t N -a resnet18 --epochs 40 --seed 0 -b 128 --log logs/mdd/dSprites_S2N

# MPI3D
CUDA_VISIBLE_DEVICES=3 python mdd.py data/mpi3d -d MPI3D -s RL -t RC -a resnet18 --epochs 60 --seed 0 -b 36 --log logs/mdd/MPI3D_RL2RC --normalization IN --resize-size 224
CUDA_VISIBLE_DEVICES=3 python mdd.py data/mpi3d -d MPI3D -s RL -t T -a resnet18 --epochs 60 --seed 0 -b 36 --log logs/mdd/MPI3D_RL2T --normalization IN --resize-size 224
CUDA_VISIBLE_DEVICES=3 python mdd.py data/mpi3d -d MPI3D -s RC -t RL -a resnet18 --epochs 60 --seed 0 -b 36 --log logs/mdd/MPI3D_RC2RL --normalization IN --resize-size 224
CUDA_VISIBLE_DEVICES=3 python mdd.py data/mpi3d -d MPI3D -s RC -t T -a resnet18 --epochs 60 --seed 0 -b 36 --log logs/mdd/MPI3D_RC2T --normalization IN --resize-size 224
CUDA_VISIBLE_DEVICES=3 python mdd.py data/mpi3d -d MPI3D -s T -t RL -a resnet18 --epochs 60 --seed 0 -b 36 --log logs/mdd/MPI3D_T2RL --normalization IN --resize-size 224
CUDA_VISIBLE_DEVICES=3 python mdd.py data/mpi3d -d MPI3D -s T -t RC -a resnet18 --epochs 60 --seed 0 -b 36 --log logs/mdd/MPI3D_T2RC --normalization IN --resize-size 224


