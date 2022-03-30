# ogb-molpcba
CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds --arch 'gin_virtual' \
    --deterministic --log logs/erm/obg --lr 3e-2 --wd 0.0 \
    --epochs 200 --metric ap -b 4096 4096 --seed 0