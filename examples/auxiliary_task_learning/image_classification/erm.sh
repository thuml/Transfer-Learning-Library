# Single Task Learning
for task in c i p q r s; do
  CUDA_VISIBLE_DEVICES=0 python erm.py data/domainnet -d DomainNetv2 -tr ${task} -ts ${task} -a resnet101 --epochs 20 \
    --seed 0 --log logs/STL/DomainNet_${task}
done

# Multi-Task Learning in a task-sampler fashion
CUDA_VISIBLE_DEVICES=0 python erm.py data/domainnet -d DomainNetv2 -tr c i p q r s -ts c i p q r s -a resnet101 \
  --epochs 20 --seed 0 --sampler UniformMultiTaskSampler --log logs/MTL/DomainNet


# Post-train
for task in c i p q r s; do
  CUDA_VISIBLE_DEVICES=6 python erm.py data/domainnet -d DomainNetv2 -tr ${task} -ts ${task} -a resnet101 --epochs 10 \
    --seed 0 --log logs/PostTrain/DomainNet_${task} --pretrained logs/MTL/DomainNet/checkpoints/best.pth
done