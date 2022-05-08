#!/usr/bin/env bash

# ImageNet Supervised Pretrain (ResNet50)
# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python noisy_student.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
  --lr 0.01 --finetune --epochs 20 --seed 0 --log logs/noisy_student/cifar100_4_labels_per_class/iter_0

for round in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python noisy_student.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
    --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
    --pretrained-teacher logs/noisy_student/cifar100_4_labels_per_class/iter_$round/checkpoints/latest.pth \
    --lr 0.01 --finetune --epochs 40 --T 0.5 --seed 0 --log logs/noisy_student/cifar100_4_labels_per_class/iter_$((round + 1))
done

# ImageNet Unsupervised Pretrain (MoCov2, ResNet50)
# ======================================================================================================================
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python noisy_student.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
  --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
  --lr 0.001 --finetune --lr-scheduler cos --epochs 20 --seed 0 \
  --log logs/noisy_student_moco_pretrain/cifar100_4_labels_per_class/iter_0

for round in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python noisy_student.py data/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
    --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
    --pretrained-backbone checkpoints/moco_v2_800ep_backbone.pth \
    --pretrained-teacher logs/noisy_student_moco_pretrain/cifar100_4_labels_per_class/iter_$round/checkpoints/latest.pth \
    --lr 0.001 --finetune --lr-scheduler cos --epochs 40 --T 1 --seed 0 \
    --log logs/noisy_student_moco_pretrain/cifar100_4_labels_per_class/iter_$((round + 1))
done
