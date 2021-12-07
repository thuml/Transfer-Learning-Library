# Faster RCNN: VOC->Clipart (~3h)
CUDA_VISIBLE_DEVICES=3 python source_only.py \
  --config-file configs/PascalVOC-Detection/faster_rcnn_R_101_C4.yaml \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 --target Clipart datasets/clipart \
  --test VOC2007Test datasets/VOC2007 Clipart datasets/clipart --finetune \
  OUTPUT_DIR logs/src_only_faster_rcnn_R_101_C4/voc2clipart

# Faster RCNN: VOC->WaterColor, Comic
CUDA_VISIBLE_DEVICES=4 python source_only.py \
  --config-file configs/PascalVOC-Detection/faster_rcnn_R_101_C4.yaml \
  -s VOC2007Partial datasets/VOC2007 VOC2012Partial datasets/VOC2012 --target WaterColor datasets/watercolor Comic datasets/comic \
  --test VOC2007PartialTest datasets/VOC2007 WaterColorTest datasets/watercolor ComicTest datasets/comic  --finetune \
  OUTPUT_DIR logs/src_only_faster_rcnn_R_101_C4/voc2watercolor_comic MODEL.ROI_HEADS.NUM_CLASSES 6

# Faster RCNN: Cityscapes->Foggy Cityscapes
CUDA_VISIBLE_DEVICES=5 python source_only.py \
  --config-file configs/Cityscapes/faster_rcnn_R_101_C4.yaml \
  -s Cityscapes datasets/cityscapes_in_voc/ --target FoggyCityscapes datasets/foggy_cityscapes_in_voc \
  --test CityscapesTest datasets/cityscapes_in_voc/ FoggyCityscapesTest datasets/foggy_cityscapes_in_voc  --finetune \
  OUTPUT_DIR logs/src_only_faster_rcnn_R_101_C4/cityscapes2foggy

# Faster RCNN: Sim10k -> Cityscapes Car
CUDA_VISIBLE_DEVICES=6 python source_only.py \
  --config-file configs/Cityscapes/faster_rcnn_R_101_C4.yaml \
  -s Sim10kCar datasets/sim10k --target CityscapesCar datasets/cityscapes_in_voc/  \
  --test CityscapesCarTest datasets/cityscapes_in_voc/ --finetune \
  OUTPUT_DIR logs/src_only_faster_rcnn_R_101_C4/sim10k2cityscapes_car MODEL.ROI_HEADS.NUM_CLASSES 1

# Faster RCNN: GTA5 -> Cityscapes
CUDA_VISIBLE_DEVICES=7 python source_only.py \
  --config-file configs/Cityscapes/faster_rcnn_R_101_C4.yaml \
  -s GTA5 datasets/synscapes_detection --target Cityscapes datasets/cityscapes_in_voc/  \
  --test CityscapesTest datasets/cityscapes_in_voc/ --finetune \
  OUTPUT_DIR logs/src_only_faster_rcnn_R_101_C4/gta52cityscapes

