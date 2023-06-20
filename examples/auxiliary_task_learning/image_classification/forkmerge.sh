# Full-version of ForkMerge: Use ForkMerge for Joint Optimization and  Task Selection Simultaneously
for task in c i p q r s; do
  CUDA_VISIBLE_DEVICES=1 python forkmerge.py data/domainnet -d DomainNetv2 -tr c i p q r s -ts ${task} -a resnet101 \
    --epochs 20 -i 2500 --seed 1 \
    --log logs/forkmerge/DomainNet_${task}
done

# Fast-version of ForkMerge: Use ForkMerge Only for Joint Optimization
for task in c i p q r s; do
  CUDA_VISIBLE_DEVICES=1 python forkmerge.py data/domainnet -d DomainNetv2 -tr c i p q r s -ts ${task} -a resnet101 \
    --epochs 20 -i 2500 --seed 1 \
    --log logs/forkmerge_fast/DomainNet_${task} --fast
done