# Image classification examples

## Fine-tune the supervised pre-trained model


## Fine-tune the unsupervised pre-trained model
Take MoCo as an example. 

1. Download MoCo pretrained checkpoints from https://github.com/facebookresearch/moco
2. Convert  the format of the MoCo checkpoints to to the standard format of pytorch
```shell
mkdir checkpoints
python convert_moco_to_pretrained.py checkpoints/moco_v1_200ep_pretrain.pth.tar checkpoints/moco_v1_200ep_backbone.pth checkpoints/moco_v1_200ep_fc.pth
```
3. Start training
```shell
CUDA_VISIBLE_DEVICES=0 python bi_tuning.py data/cub200 -d CUB200 -sr 100 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_bi_tuning/cub200_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth
```
