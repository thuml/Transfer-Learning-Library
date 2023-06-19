# STL / single task learning
for task in segmentation depth normal; do
  CUDA_VISIBLE_DEVICES=1 python erm.py data/nyuv2 -tr ${task} -ts ${task} --log logs/STL/${task}
done

# MTL Equal Weight
CUDA_VISIBLE_DEVICES=1 python erm.py data/nyuv2 -tr segmentation depth normal -ts segmentation depth normal --log logs/EW
