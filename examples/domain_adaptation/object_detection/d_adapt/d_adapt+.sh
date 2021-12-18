# D-adapt+: Decouple Adaptation with category adaptor and bounding box adaptor
# ResNet101 Based Faster RCNN: Faster RCNN: VOC->Clipart
# 44.8
pretrained_models=../source_only/logs/faster_rcnn_R_101_C4/voc2clipart/model_final.pth
CUDA_VISIBLE_DEVICES=7 python d_adapt.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_R_101_C4/voc2clipart_v3/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 47.9
pretrained_models=logs/d_adapt+_faster_rcnn_R_101_C4/voc2clipart_v3/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=7 python d_adapt.py --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_R_101_C4/voc2clipart_v3/phase2 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 49.0
pretrained_models=logs/d_adapt+_faster_rcnn_R_101_C4/voc2clipart_v3/phase2/model_final.pth
CUDA_VISIBLE_DEVICES=7 python d_adapt.py --confidence-ratio-c 0.2 \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_R_101_C4/voc2clipart_v3/phase3 MODEL.WEIGHTS ${pretrained_models} SEED 0

# ResNet101 Based Faster RCNN: Faster RCNN: VOC->Comic
# 39.7
pretrained_models=../source_only/logs/faster_rcnn_R_101_C4/voc2watercolor_comic/model_final.pth
CUDA_VISIBLE_DEVICES=7 python d_adapt.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t Comic ../datasets/comic --test ComicTest ../datasets/comic --finetune --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_R_101_C4/voc2comic_v3/phase1 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 41.0
pretrained_models=logs/d_adapt+_faster_rcnn_R_101_C4/voc2comic_v3/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=7 python d_adapt.py --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t Comic ../datasets/comic --test ComicTest ../datasets/comic --finetune --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_R_101_C4/voc2comic_v3/phase2 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0

# ResNet101 Based Faster RCNN: Sim10k -> Cityscapes Car
# 51.9
pretrained_models=../source_only/logs/faster_rcnn_R_101_C4/sim10k2cityscapes_car/model_final.pth
CUDA_VISIBLE_DEVICES=5 python d_adapt.py --workers-c 8 --ignored-scores-c 0.05 0.5 --bottleneck-dim-c 256 --bottleneck-dim-b 256 \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Sim10kCar ../datasets/sim10k -t CityscapesCar ../datasets/cityscapes_in_voc/  \
  --test CityscapesCarTest ../datasets/cityscapes_in_voc/ --finetune --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_R_101_C4/sim10k2cityscapes_car_v5/phase1 MODEL.ROI_HEADS.NUM_CLASSES 1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 52.0
#pretrained_models=../source_only/logs/faster_rcnn_R_101_C4/sim10k2cityscapes_car/model_final.pth
#CUDA_VISIBLE_DEVICES=5 python d_adapt.py --workers-c 8 --ignored-scores-c 0.05 0.5 --bottleneck-dim-c 256 --bottleneck-dim-b 256  --confidence-ratio-c 0.5 \
#  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
#  -s Sim10kCar ../datasets/sim10k -t CityscapesCar ../datasets/cityscapes_in_voc/  \
#  --test CityscapesCarTest ../datasets/cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
#  OUTPUT_DIR logs/d_adapt+_faster_rcnn_R_101_C4/sim10k2cityscapes_car_v6/phase1 MODEL.ROI_HEADS.NUM_CLASSES 1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# ResNet101 Based Faster RCNN: Cityscapes -> Foggy Cityscapes
# 40.1
pretrained_models=../source_only/logs/faster_rcnn_R_101_C4/cityscapes2foggy/model_final.pth
CUDA_VISIBLE_DEVICES=6 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_R_101_C4/cityscapes2foggy_v3/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 42.4
pretrained_models=logs/d_adapt+_faster_rcnn_R_101_C4/cityscapes2foggy_v3/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=6 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_R_101_C4/cityscapes2foggy_v3/phase2 MODEL.WEIGHTS ${pretrained_models} SEED 0


# VGG Based Faster RCNN: Cityscapes -> Foggy Cityscapes
# 33.3
pretrained_models=../source_only/logs/faster_rcnn_vgg_16/cityscapes2foggy/model_final.pth
CUDA_VISIBLE_DEVICES=4 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_vgg_16/cityscapes2foggy_v3/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 37.0
pretrained_models=logs/d_adapt+_faster_rcnn_vgg_16/cityscapes2foggy_v3/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=4 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_vgg_16/cityscapes2foggy_v3/phase2 MODEL.WEIGHTS ${pretrained_models} SEED 0

#  38.9
pretrained_models=logs/d_adapt+_faster_rcnn_vgg_16/cityscapes2foggy_v3/phase2/model_final.pth
CUDA_VISIBLE_DEVICES=4 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 --confidence-ratio-c 0.2 \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_vgg_16/cityscapes2foggy_v3/phase3 MODEL.WEIGHTS ${pretrained_models} SEED 0

# VGG Based Faster RCNN: Sim10k -> Cityscapes Car
# 49.3
pretrained_models=../source_only/logs/faster_rcnn_vgg_16/sim10k2cityscapes_car/model_final.pth
CUDA_VISIBLE_DEVICES=5 python d_adapt.py --workers-c 8 --ignored-scores-c 0.05 0.5 --bottleneck-dim-c 256 --bottleneck-dim-b 256 \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Sim10kCar ../datasets/sim10k -t CityscapesCar ../datasets/cityscapes_in_voc/  \
  --test CityscapesCarTest ../datasets/cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/d_adapt+_faster_rcnn_vgg_16/sim10k2cityscapes_car_v3/phase1 MODEL.ROI_HEADS.NUM_CLASSES 1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# RetinaNet: VOC->Clipart
# 41.2

#
#pretrained_models=../logs/source_only/retinanet_R_101_FPN/voc2clipart/model_final.pth
#CUDA_VISIBLE_DEVICES=3 python d_adapt.py --ignored-ious-c 0.3 0.7 \
#  --config-file config/retinanet_R_101_FPN_voc.yaml \
#  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
#  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
#  --finetune \
#  OUTPUT_DIR logs/d_adapt+_retinanet_R_101_FPN/voc2clipart_v3/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

#pretrained_models=logs/d_adapt+_retinanet_R_101_FPN/voc2clipart_v3/phase1/model_final.pth
#CUDA_VISIBLE_DEVICES=3 python d_adapt.py --confidence-ratio-c 0.1 \
#  --config-file config/retinanet_R_101_FPN_voc.yaml \
#  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
#  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
#  --finetune --bbox-refine \
#  OUTPUT_DIR logs/d_adapt+_retinanet_R_101_FPN/voc2clipart_v3/phase2 MODEL.WEIGHTS ${pretrained_models} SEED 0
#
#pretrained_models=../logs/source_only/retinanet_R_101_FPN/voc2clipart/model_final.pth
#CUDA_VISIBLE_DEVICES=1 python d_adapt.py \
#  --config-file config/retinanet_R_101_FPN_voc.yaml \
#  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
#  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
#  --finetune -- \
#  OUTPUT_DIR logs/d_adapt+_retinanet_R_101_FPN/voc2clipart_v5/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

pretrained_models=../logs/source_only/retinanet_R_101_FPN/voc2clipart/model_final.pth
CUDA_VISIBLE_DEVICES=3 python d_adapt.py --batch-size-c 32 \
  --config-file config/retinanet_R_101_FPN_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune \
  OUTPUT_DIR logs/d_adapt+_retinanet_R_101_FPN/voc2clipart_v6/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0


pretrained_models=../logs/source_only/retinanet_R_101_FPN/voc2clipart/model_final.pth
CUDA_VISIBLE_DEVICES=4 python d_adapt.py --batch-size-c 32 --resize-size-c 224 \
  --config-file config/retinanet_R_101_FPN_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune \
  OUTPUT_DIR logs/d_adapt+_retinanet_R_101_FPN/voc2clipart_v7/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0


