# D-adapt: Decouple Adaptation with only category adaptor
# ResNet101 Based Faster RCNN: Faster RCNN: VOC->Clipart
pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/voc2clipart/model_final.pth
CUDA_VISIBLE_DEVICES=4 python d_adapt.py \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart \
  --test Clipart ../datasets/clipart --finetune \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2clipart/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

pretrained_models=logs/faster_rcnn_R_101_C4/voc2clipart/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=4 python d_adapt.py --confidence-ratio-c 0.1 \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart \
  --test Clipart ../datasets/clipart --finetune \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2clipart/phase2 MODEL.WEIGHTS ${pretrained_models} SEED 0

pretrained_models=logs/faster_rcnn_R_101_C4/voc2clipart/phase2/model_final.pth
CUDA_VISIBLE_DEVICES=4 python d_adapt.py --confidence-ratio-c 0.2 \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart \
  --test Clipart ../datasets/clipart --finetune \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2clipart/phase3 MODEL.WEIGHTS ${pretrained_models} SEED 0


# ResNet101 Based Faster RCNN: Faster RCNN: VOC->Comic
pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/voc2watercolor_comic/model_final.pth
CUDA_VISIBLE_DEVICES=5 python d_adapt.py \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t Comic ../datasets/comic \
  --test ComicTest ../datasets/comic --finetune \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2comic/phase1 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0

pretrained_models=logs/faster_rcnn_R_101_C4/voc2comic/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=5 python d_adapt.py --confidence-ratio-c 0.1 \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t Comic ../datasets/comic \
  --test ComicTest ../datasets/comic --finetune \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2comic/phase2 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0

pretrained_models=logs/faster_rcnn_R_101_C4/voc2comic/phase2/model_final.pth
CUDA_VISIBLE_DEVICES=5 python d_adapt.py --confidence-ratio-c 0.2 \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t Comic ../datasets/comic \
  --test ComicTest ../datasets/comic --finetune \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2comic/phase3 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0

# ResNet101 Based Faster RCNN: Cityscapes -> Foggy Cityscapes 38.6
pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/cityscapes2foggy/model_final.pth
CUDA_VISIBLE_DEVICES=2 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 \
  OUTPUT_DIR logs/d_adapt_faster_rcnn_R_101_C4/cityscapes2foggy_v2/phase1 MODEL.WEIGHTS ${pretrained_models} SEED 0

# 41.4
pretrained_models=logs/d_adapt_faster_rcnn_R_101_C4/cityscapes2foggy_v2/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=2 python d_adapt.py --workers-c 4 --max-train-c 20 --ignored-scores-c 0.05 0.5 --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Cityscapes ../datasets/cityscapes_in_voc -t FoggyCityscapes ../datasets/foggy_cityscapes_in_voc/  \
  --test FoggyCityscapesTest ../datasets/foggy_cityscapes_in_voc/ --finetune --trade-off 0.5 \
  OUTPUT_DIR logs/d_adapt_faster_rcnn_R_101_C4/cityscapes2foggy_v2/phase2 MODEL.WEIGHTS ${pretrained_models} SEED 0

