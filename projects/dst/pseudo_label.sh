# ======================================================================================================================
# CIFAR10 WideResNet-28-2 40 labels
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py --root data/cifar10 --dataset CIFAR10 --num-samples-per-class 4 \
  --norm-mean 0.491 0.482 0.447 --norm-std 0.247 0.244 0.262 \
  --arch WideResNet --depth 28 --widen-factor 2 --lr 0.03 -b 64 -ub 64 --threshold 0.95 --weight-decay 0.0005 \
  --opt-level O0 --log logs/pseudo_label/cifar10_40_labels
# ======================================================================================================================

# ======================================================================================================================
# CIFAR100 WideResNet-28-8 400 labels
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py --root data/cifar100 --dataset CIFAR100 --num-samples-per-class 4 \
  --norm-mean 0.507 0.487 0.441 --norm-std 0.267 0.256 0.276 \
  --arch WideResNet --depth 28 --widen-factor 8 --lr 0.03 -b 64 -ub 64 --threshold 0.95 --weight-decay 0.001 \
  --opt-level O0 --log logs/pseudo_label/cifar100_400_labels
# ======================================================================================================================

# ======================================================================================================================
# SVHN WideResNet-28-2 40 labels
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py --root data/svhn --dataset SVHN --num-samples-per-class 4 \
  --norm-mean 0.438 0.444 0.473 --norm-std 0.175 0.177 0.174 \
  --arch WideResNet --depth 28 --widen-factor 2 --lr 0.03 -b 64 -ub 64 --threshold 0.95 --weight-decay 0.0005 \
  --opt-level O0 --log logs/pseudo_label/svhn_40_labels
# ======================================================================================================================

# ======================================================================================================================
# STL10 WideResNetVar-28-2 (WideResNet-37-2 in FixMatch) 40 labels
CUDA_VISIBLE_DEVICES=0 python pseudo_label.py --root data/stl10 --dataset STL10 --num-samples-per-class 4 \
  --img-size 96 96 --norm-mean 0.441 0.428 0.387 --norm-std 0.268 0.261 0.269 \
  --arch WideResNetVar --depth 28 --widen-factor 2 --lr 0.03 -b 64 -ub 64 --threshold 0.95 --weight-decay 0.0005 \
  --opt-level O0 --log logs/pseudo_label/stl10_40_labels
# ======================================================================================================================
