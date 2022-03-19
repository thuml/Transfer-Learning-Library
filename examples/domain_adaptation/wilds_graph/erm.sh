# ogb-molpcba
CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d 'ogb-molpcba' --arch 'gin_virtual' \
    --deterministic --log logs/erm/obg --lr 2e-2 --wd 0.0 \
    --epochs 200 --metric ap -b 4096 4096 --seed 0