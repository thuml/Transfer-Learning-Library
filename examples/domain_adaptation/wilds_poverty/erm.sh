# official split scheme
CUDA_VISIBLE_DEVICES=0 python erm.py data/wilds --split-scheme official --fold A \
  --arch 'resnet18_ms' --lr 1e-3 --epochs 200 -b 64 64 --opt-level O1 --deterministic --log logs/erm/poverty_fold_A
CUDA_VISIBLE_DEVICES=0 python erm.py data/wilds --split-scheme official --fold B \
  --arch 'resnet18_ms' --lr 1e-3 --epochs 200 -b 64 64 --opt-level O1 --deterministic --log logs/erm/poverty_fold_B
CUDA_VISIBLE_DEVICES=0 python erm.py data/wilds --split-scheme official --fold C \
  --arch 'resnet18_ms' --lr 1e-3 --epochs 200 -b 64 64 --opt-level O1 --deterministic --log logs/erm/poverty_fold_C
CUDA_VISIBLE_DEVICES=0 python erm.py data/wilds --split-scheme official --fold D \
  --arch 'resnet18_ms' --lr 1e-3 --epochs 200 -b 64 64 --opt-level O1 --deterministic --log logs/erm/poverty_fold_D
CUDA_VISIBLE_DEVICES=0 python erm.py data/wilds --split-scheme official --fold E \
  --arch 'resnet18_ms' --lr 1e-3 --epochs 200 -b 64 64 --opt-level O1 --deterministic --log logs/erm/poverty_fold_E
