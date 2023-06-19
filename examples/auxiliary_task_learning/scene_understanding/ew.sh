
# MTL Equal Weight
CUDA_VISIBLE_DEVICES=0 python ew.py data/nyuv2 -tr segmentation depth normal -ts segmentation depth normal --log logs/EW_v2 -b 8
