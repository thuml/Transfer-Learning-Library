# DSprites
CUDA_VISIBLE_DEVICES=0 python rsd.py data/dSprites -d DSprites -s C -t N -a resnet18 --epochs 40 --seed 0 --log logs/rsd/DSprites_C2N
CUDA_VISIBLE_DEVICES=0 python rsd.py data/dSprites -d DSprites -s C -t S -a resnet18 --epochs 40 --seed 0 --log logs/rsd/DSprites_C2S
CUDA_VISIBLE_DEVICES=0 python rsd.py data/dSprites -d DSprites -s N -t C -a resnet18 --epochs 40 --seed 0 --log logs/rsd/DSprites_N2C
CUDA_VISIBLE_DEVICES=0 python rsd.py data/dSprites -d DSprites -s N -t S -a resnet18 --epochs 40 --seed 0 --log logs/rsd/DSprites_N2S
CUDA_VISIBLE_DEVICES=0 python rsd.py data/dSprites -d DSprites -s S -t C -a resnet18 --epochs 40 --seed 0 --log logs/rsd/DSprites_S2C
CUDA_VISIBLE_DEVICES=0 python rsd.py data/dSprites -d DSprites -s S -t N -a resnet18 --epochs 40 --seed 0 --log logs/rsd/DSprites_S2N

# MPI3D
CUDA_VISIBLE_DEVICES=0 python rsd.py data/mpi3d -d MPI3D -s RL -t RC -a resnet18 --epochs 40 --seed 0 --log logs/rsd/MPI3D_RL2RC --resize-size 224
CUDA_VISIBLE_DEVICES=0 python rsd.py data/mpi3d -d MPI3D -s RL -t T -a resnet18 --epochs 40 --seed 0 --log logs/rsd/MPI3D_RL2T --resize-size 224
CUDA_VISIBLE_DEVICES=0 python rsd.py data/mpi3d -d MPI3D -s RC -t RL -a resnet18 --epochs 40 --seed 0 --log logs/rsd/MPI3D_RC2RL --resize-size 224
CUDA_VISIBLE_DEVICES=0 python rsd.py data/mpi3d -d MPI3D -s RC -t T -a resnet18 --epochs 40 --seed 0 --log logs/rsd/MPI3D_RC2T --resize-size 224
CUDA_VISIBLE_DEVICES=0 python rsd.py data/mpi3d -d MPI3D -s T -t RL -a resnet18 --epochs 40 --seed 0 --log logs/rsd/MPI3D_T2RL --resize-size 224
CUDA_VISIBLE_DEVICES=0 python rsd.py data/mpi3d -d MPI3D -s T -t RC -a resnet18 --epochs 40 --seed 0 --log logs/rsd/MPI3D_T2RC --resize-size 224
