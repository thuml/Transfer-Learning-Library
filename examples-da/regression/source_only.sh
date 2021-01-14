# DSprites
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/dSprites -d DSprites -s C -t N -a resnet18  --epochs 20 --seed 0 > benchmarks/da/regression/source_only/DSprites_C2N.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/dSprites -d DSprites -s C -t S -a resnet18  --epochs 20 --seed 0 > benchmarks/da/regression/source_only/DSprites_C2S.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/dSprites -d DSprites -s N -t C -a resnet18  --epochs 20 --seed 0 > benchmarks/da/regression/source_only/DSprites_N2C.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/dSprites -d DSprites -s N -t S -a resnet18  --epochs 20 --seed 0 > benchmarks/da/regression/source_only/DSprites_N2S.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/dSprites -d DSprites -s S -t C -a resnet18  --epochs 20 --seed 0 > benchmarks/da/regression/source_only/DSprites_S2C.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/dSprites -d DSprites -s S -t N -a resnet18  --epochs 20 --seed 0 > benchmarks/da/regression/source_only/DSprites_S2N.txt

# MPI3D
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/mpi3d -d MPI3D -s RL -t RC -a resnet18  --epochs 40 --seed 0 -b 128 > benchmarks/da/regression/source_only/MPI3D_RL2RC.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/mpi3d -d MPI3D -s RL -t T -a resnet18  --epochs 40 --seed 0 -b 128 > benchmarks/da/regression/source_only/MPI3D_RL2T.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/mpi3d -d MPI3D -s RC -t RL -a resnet18  --epochs 40 --seed 0 -b 128 > benchmarks/da/regression/source_only/MPI3D_RC2RL.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/mpi3d -d MPI3D -s RC -t T -a resnet18  --epochs 40 --seed 0 -b 128 > benchmarks/da/regression/source_only/MPI3D_RC2T.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/mpi3d -d MPI3D -s T -t RL -a resnet18  --epochs 40 --seed 0 -b 128 > benchmarks/da/regression/source_only/MPI3D_T2RL.txt
CUDA_VISIBLE_DEVICES=0 python examples-da/regression/source_only.py data/mpi3d -d MPI3D -s T -t RC -a resnet18  --epochs 40 --seed 0 -b 128 > benchmarks/da/regression/source_only/MPI3D_T2RC.txt
