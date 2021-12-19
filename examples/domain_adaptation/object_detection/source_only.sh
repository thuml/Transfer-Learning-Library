# Faster RCNN: VOC->Clipart
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 -t Clipart datasets/clipart \
  --test VOC2007Test datasets/VOC2007 Clipart datasets/clipart --finetune \
  OUTPUT_DIR logs/source_only/faster_rcnn_R_101_C4/voc2clipart

# Faster RCNN: VOC->WaterColor, Comic
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007Partial datasets/VOC2007 VOC2012Partial datasets/VOC2012 -t WaterColor datasets/watercolor Comic datasets/comic \
  --test VOC2007PartialTest datasets/VOC2007 WaterColorTest datasets/watercolor ComicTest datasets/comic  --finetune \
  OUTPUT_DIR logs/source_only/faster_rcnn_R_101_C4/voc2watercolor_comic MODEL.ROI_HEADS.NUM_CLASSES 6

# ResNet101 Based Faster RCNN: Cityscapes->Foggy Cityscapes
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Cityscapes datasets/cityscapes_in_voc/ -t FoggyCityscapes datasets/foggy_cityscapes_in_voc \
  --test CityscapesTest datasets/cityscapes_in_voc/ FoggyCityscapesTest datasets/foggy_cityscapes_in_voc --finetune \
  OUTPUT_DIR logs/source_only/faster_rcnn_R_101_C4/cityscapes2foggy

# VGG16 Based Faster RCNN: Cityscapes->Foggy Cityscapes
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Cityscapes datasets/cityscapes_in_voc/ -t FoggyCityscapes datasets/foggy_cityscapes_in_voc \
  --test CityscapesTest datasets/cityscapes_in_voc/ FoggyCityscapesTest datasets/foggy_cityscapes_in_voc --finetune \
  OUTPUT_DIR logs/source_only/faster_rcnn_vgg_16/cityscapes2foggy

# ResNet101 Based Faster RCNN: Sim10k -> Cityscapes Car
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Sim10kCar datasets/sim10k -t CityscapesCar datasets/cityscapes_in_voc/  \
  --test CityscapesCarTest datasets/cityscapes_in_voc/ --finetune \
  OUTPUT_DIR logs/source_only/faster_rcnn_R_101_C4/sim10k2cityscapes_car MODEL.ROI_HEADS.NUM_CLASSES 1

# VGG16 Based Faster RCNN: Sim10k -> Cityscapes Car
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Sim10kCar datasets/sim10k -t CityscapesCar datasets/cityscapes_in_voc/  \
  --test CityscapesCarTest datasets/cityscapes_in_voc/ --finetune \
  OUTPUT_DIR logs/source_only/faster_rcnn_vgg_16/sim10k2cityscapes_car MODEL.ROI_HEADS.NUM_CLASSES 1

# Faster RCNN: GTA5 -> Cityscapes
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s GTA5 datasets/synscapes_detection -t Cityscapes datasets/cityscapes_in_voc/  \
  --test CityscapesTest datasets/cityscapes_in_voc/ --finetune \
  OUTPUT_DIR logs/source_only/faster_rcnn_R_101_C4/gta52cityscapes

# RetinaNet: VOC->Clipart
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/retinanet_R_101_FPN_voc.yaml \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 -t Clipart datasets/clipart \
  --test VOC2007Test datasets/VOC2007 Clipart datasets/clipart --finetune \
  OUTPUT_DIR logs/source_only/retinanet_R_101_FPN/voc2clipart

# RetinaNet: VOC->WaterColor, Comic
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/retinanet_R_101_FPN_voc.yaml \
  -s VOC2007Partial datasets/VOC2007 VOC2012Partial datasets/VOC2012 -t WaterColor datasets/watercolor Comic datasets/comic \
  --test VOC2007PartialTest datasets/VOC2007 WaterColorTest datasets/watercolor ComicTest datasets/comic --finetune \
  OUTPUT_DIR logs/source_only/retinanet_R_101_FPN/voc2watercolor_comic MODEL.RETINANET.NUM_CLASSES 6
