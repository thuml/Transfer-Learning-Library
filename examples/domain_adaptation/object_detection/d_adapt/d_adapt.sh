# D-adapt: Decouple Adaptation with only category adaptor
# Faster RCNN: VOC->Clipart
pretrained_models=../source_only/logs/faster_rcnn_R_101_C4/voc2clipart/model_final.pth
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

pretrained_models=../source_only/logs/faster_rcnn_R_101_C4/voc2clipart/phase2/model_final.pth
CUDA_VISIBLE_DEVICES=4 python d_adapt.py --confidence-ratio-c 0.2 \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart \
  --test Clipart ../datasets/clipart --finetune \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2clipart/phase3 MODEL.WEIGHTS ${pretrained_models} SEED 0

# Faster RCNN: VOC->Comic
pretrained_models=../source_only/logs/faster_rcnn_R_101_C4/voc2watercolor_comic/model_final.pth
CUDA_VISIBLE_DEVICES=5 python d_adapt.py \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t Comic ../datasets/comic \
  --test ComicTest ../datasets/comic --finetune \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2comic/phase1 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0

pretrained_models=logs/faster_rcnn_R_101_C4/voc2watercolor_comic/phase1/model_final.pth
CUDA_VISIBLE_DEVICES=5 python d_adapt.py --confidence-ratio-c 0.1 \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t Comic ../datasets/comic \
  --test ComicTest ../datasets/comic --finetune \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2comic/phase2 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0

pretrained_models=logs/faster_rcnn_R_101_C4/voc2watercolor_comic/phase2/model_final.pth
CUDA_VISIBLE_DEVICES=5 python d_adapt.py --confidence-ratio-c 0.2 \
  -s VOC2007Partial ../datasets/VOC2007 VOC2012Partial ../datasets/VOC2012  \
  -t Comic ../datasets/comic \
  --test ComicTest ../datasets/comic --finetune \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2comic/phase3 MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS ${pretrained_models} SEED 0
