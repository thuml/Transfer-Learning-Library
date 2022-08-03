# Faster RCNN: WaterColor
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s WaterColor datasets/watercolor -t WaterColor datasets/watercolor \
  --test WaterColorTest datasets/watercolor  --finetune \
  OUTPUT_DIR logs/oracle/faster_rcnn_R_101_C4/watercolor MODEL.ROI_HEADS.NUM_CLASSES 6

# Faster RCNN: Comic
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s Comic datasets/comic -t Comic datasets/comic \
  --test ComicTest datasets/comic  --finetune \
  OUTPUT_DIR logs/oracle/faster_rcnn_R_101_C4/comic MODEL.ROI_HEADS.NUM_CLASSES 6

# ResNet101 Based Faster RCNN: Cityscapes->Foggy Cityscapes
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s FoggyCityscapes datasets/foggy_cityscapes_in_voc -t FoggyCityscapes datasets/foggy_cityscapes_in_voc \
  --test FoggyCityscapesTest datasets/foggy_cityscapes_in_voc --finetune \
  OUTPUT_DIR logs/oracle/faster_rcnn_R_101_C4/cityscapes2foggy

# VGG16 Based Faster RCNN: Cityscapes->Foggy Cityscapes
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s FoggyCityscapes datasets/foggy_cityscapes_in_voc -t FoggyCityscapes datasets/foggy_cityscapes_in_voc \
  --test FoggyCityscapesTest datasets/foggy_cityscapes_in_voc --finetune \
  OUTPUT_DIR logs/oracle/faster_rcnn_vgg_16/cityscapes2foggy

# ResNet101 Based Faster RCNN: Sim10k -> Cityscapes Car
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s CityscapesCar datasets/cityscapes_in_voc/ -t CityscapesCar datasets/cityscapes_in_voc/  \
  --test CityscapesCarTest datasets/cityscapes_in_voc/ --finetune \
  OUTPUT_DIR logs/oracle/faster_rcnn_R_101_C4/cityscapes_car MODEL.ROI_HEADS.NUM_CLASSES 1

# VGG16 Based Faster RCNN: Sim10k -> Cityscapes Car
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
 -s CityscapesCar datasets/cityscapes_in_voc/ -t CityscapesCar datasets/cityscapes_in_voc/  \
  --test CityscapesCarTest datasets/cityscapes_in_voc/ --finetune \
  OUTPUT_DIR logs/oracle/faster_rcnn_vgg_16/cityscapes_car MODEL.ROI_HEADS.NUM_CLASSES 1

