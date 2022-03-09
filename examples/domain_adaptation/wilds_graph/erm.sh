# ogb-molpcba
CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds -d 'ogb-molpcba' --arch 'gin-virtual' \
    --opt-level O1 --deterministic --log logs/erm/obg/test --lr 1e-3 --wd 0.0 \
    --epochs 100 --metric ap -b 32 32 --seed 0

CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds -d 'ogb-molpcba' --arch 'gin-virtual' \
    --opt-level O1 --deterministic --log logs/erm/obg/test --lr 1e-3 --wd 0.0 \
    --epochs 100 --metric ap -b 32 32 --seed 1

CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d 'ogb-molpcba' --arch 'gin-virtual' \
    --opt-level O1 --deterministic --log logs/erm/obg/test --lr 1e-3 --wd 0.0 \
    --epochs 100 --metric ap -b 32 32 --seed 2

CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d 'ogb-molpcba' --arch 'gin-virtual' \
    --opt-level O1 --deterministic --log logs/erm/obg/test --lr 1e-3 --wd 0.0 \
    --epochs 100 --metric ap -b 32 32 --seed 3