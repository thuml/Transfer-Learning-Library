# ResNet101 Based Faster RCNN: Faster RCNN: VOC->Clipart
# 44.8
pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/voc2clipart/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2clipart/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 47.9
pretrained_models=logs/faster_rcnn_R_101_C4/voc2clipart/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2clipart/phase2 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 49.0
pretrained_models=logs/faster_rcnn_R_101_C4/voc2clipart/phase2/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --confidence-ratio-c 0.2 \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2clipart/phase3 MODEL.WEIGHTS ${pretrained_models} SEED 0

# ResNet101 Based Faster RCNN: Faster RCNN: VOC->WaterColor
# 54.1
pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/voc2watercolor_comic/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t WaterColor ../datasets/watercolor --test WaterColorTest ../datasets/watercolor --finetune --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2watercolor/phase1 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 57.5
pretrained_models=logs/faster_rcnn_R_101_C4/voc2watercolor/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t WaterColor ../datasets/watercolor --test WaterColorTest ../datasets/watercolor --finetune --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2watercolor/phase2 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0

# ResNet101 Based Faster RCNN: Faster RCNN: VOC->Comic
# 39.7
pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/voc2watercolor_comic/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t Comic ../datasets/comic --test ComicTest ../datasets/comic --finetune --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2comic/phase1 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 41.0
pretrained_models=logs/faster_rcnn_R_101_C4/voc2comic/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t Comic ../datasets/comic --test ComicTest ../datasets/comic --finetune --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2comic/phase2 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0

# ResNet101 Based Faster RCNN: Cityscapes -> Foggy Cityscapes
# 40.1
pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/cityscapes2foggy/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 42.4
pretrained_models=logs/faster_rcnn_R_101_C4/cityscapes2foggy/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/cityscapes2foggy/phase2 MODEL.WEIGHTS ${pretrained_models} SEED 0


# VGG Based Faster RCNN: Cityscapes -> Foggy Cityscapes
# 33.3
pretrained_models=../logs/source_only/faster_rcnn_vgg_16/cityscapes2foggy/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_vgg_16/cityscapes2foggy/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 37.0
pretrained_models=logs/faster_rcnn_vgg_16/cityscapes2foggy/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_vgg_16/cityscapes2foggy/phase2 MODEL.WEIGHTS ${pretrained_models} SEED 0

#  38.9
pretrained_models=logs/faster_rcnn_vgg_16/cityscapes2foggy/phase2/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 --confidence-ratio-c 0.2 \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_vgg_16/cityscapes2foggy/phase3 MODEL.WEIGHTS ${pretrained_models} SEED 0

# ResNet101 Based Faster RCNN: Sim10k -> Cityscapes Car
# 51.9
pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/sim10k2cityscapes_car/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --workers-c 8 --ignored-scores-c 0.05 0.5 --bottleneck-dim-c 256 --bottleneck-dim-b 256 \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Sim10kCar ../datasets/sim10k -t CityscapesCar ../datasets/cityscapes_in_voc/  \
  --test CityscapesCarTest ../datasets/cityscapes_in_voc/ --finetune --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/sim10k2cityscapes_car/phase1 MODEL.ROI_HEADS.NUM_CLASSES 1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# VGG Based Faster RCNN: Sim10k -> Cityscapes Car
# 49.3
pretrained_models=../logs/source_only/faster_rcnn_vgg_16/sim10k2cityscapes_car/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --workers-c 8 --ignored-scores-c 0.05 0.5 --bottleneck-dim-c 256 --bottleneck-dim-b 256 \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Sim10kCar ../datasets/sim10k -t CityscapesCar ../datasets/cityscapes_in_voc/  \
  --test CityscapesCarTest ../datasets/cityscapes_in_voc/ --finetune --trade-off 0.5 --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_vgg_16/sim10k2cityscapes_car/phase1 MODEL.ROI_HEADS.NUM_CLASSES 1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# RetinaNet: VOC->Clipart
# 44.7
pretrained_models=../logs/source_only/retinanet_R_101_FPN/voc2clipart/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --remove-bg \
  --config-file config/retinanet_R_101_FPN_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune --bbox-refine \
  OUTPUT_DIR logs/retinanet_R_101_FPN/voc2clipart/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 46.3
pretrained_models=logs/retinanet_R_101_FPN/voc2clipart/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --remove-bg --confidence-ratio 0.1 \
  --config-file config/retinanet_R_101_FPN_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune --bbox-refine \
  OUTPUT_DIR logs/retinanet_R_101_FPN/voc2clipart/phase2 MODEL.WEIGHTS ${pretrained_models} SEED 0
