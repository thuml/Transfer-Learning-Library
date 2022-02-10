# VOC to Clipart
mkdir datasets/VOC2007_to_clipart
cp -r datasets/VOC2007/* datasets/VOC2007_to_clipart
mkdir datasets/VOC2012_to_clipart
cp -r datasets/VOC2012/* datasets/VOC2012_to_clipart

CUDA_VISIBLE_DEVICES=0 python cycle_gan.py \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 -t Clipart datasets/clipart \
  --translated-source datasets/VOC2007_to_clipart datasets/VOC2012_to_clipart \
  --log logs/cyclegan_resnet9/translation/voc2clipart --netG resnet_9

CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 VOC2007 datasets/VOC2007_to_clipart VOC2012 datasets/VOC2012_to_clipart \
  -t Clipart datasets/clipart \
  --test VOC2007Test datasets/VOC2007 Clipart datasets/clipart --finetune \
  OUTPUT_DIR logs/cyclegan_resnet9/faster_rcnn_R_101_C4/voc2clipart

# VOC to Comic
mkdir datasets/VOC2007_to_comic
cp -r datasets/VOC2007/* datasets/VOC2007_to_comic
mkdir datasets/VOC2012_to_comic
cp -r datasets/VOC2012/* datasets/VOC2012_to_comic

CUDA_VISIBLE_DEVICES=0 python cycle_gan.py \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 -t Comic datasets/comic \
  --translated-source datasets/VOC2007_to_comic datasets/VOC2012_to_comic \
  --log logs/cyclegan_resnet9/translation/voc2comic --netG resnet_9

CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007Partial datasets/VOC2007 VOC2012Partial datasets/VOC2012 VOC2007Partial datasets/VOC2007_to_comic VOC2012Partial datasets/VOC2012_to_comic \
  -t Comic datasets/comic \
  --test VOC2007Test datasets/VOC2007 ComicTest datasets/comic --finetune \
  OUTPUT_DIR logs/cyclegan_resnet9/faster_rcnn_R_101_C4/voc2comic MODEL.ROI_HEADS.NUM_CLASSES 6

# VOC to WaterColor
mkdir datasets/VOC2007_to_watercolor
cp -r datasets/VOC2007/* datasets/VOC2007_to_watercolor
mkdir datasets/VOC2012_to_watercolor
cp -r datasets/VOC2012/* datasets/VOC2012_to_watercolor

CUDA_VISIBLE_DEVICES=0 python cycle_gan.py \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 -t WaterColor datasets/watercolor \
  --translated-source datasets/VOC2007_to_watercolor datasets/VOC2012_to_watercolor \
  --log logs/cyclegan_resnet9/translation/voc2watercolor --netG resnet_9

CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007Partial datasets/VOC2007 VOC2012Partial datasets/VOC2012 VOC2007Partial datasets/VOC2007_to_watercolor VOC2012Partial datasets/VOC2012_to_watercolor \
  -t WaterColor datasets/watercolor \
  --test VOC2007Test datasets/VOC2007 WaterColorTest datasets/watercolor --finetune \
  OUTPUT_DIR logs/cyclegan_resnet9/faster_rcnn_R_101_C4/voc2watercolor MODEL.ROI_HEADS.NUM_CLASSES 6

# Cityscapes to Foggy Cityscapes
mkdir datasets/cityscapes_to_foggy_cityscapes
cp -r datasets/cityscapes_in_voc/* datasets/cityscapes_to_foggy_cityscapes

CUDA_VISIBLE_DEVICES=0 python cycle_gan.py -s Cityscapes datasets/cityscapes_in_voc \
  -t FoggyCityscapes datasets/foggy_cityscapes_in_voc \
  --translated-source datasets/cityscapes_to_foggy_cityscapes \
  --log logs/cyclegan/translation/cityscapes2foggy

# ResNet101 Based Faster RCNN: Cityscapes->Foggy Cityscapes
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Cityscapes datasets/cityscapes_in_voc/ Cityscapes datasets/cityscapes_to_foggy_cityscapes/ \
  -t FoggyCityscapes datasets/foggy_cityscapes_in_voc \
  --test CityscapesTest datasets/cityscapes_in_voc/ FoggyCityscapesTest datasets/foggy_cityscapes_in_voc --finetune \
  OUTPUT_DIR logs/cyclegan/faster_rcnn_R_101_C4/cityscapes2foggy

# VGG16 Based Faster RCNN: Cityscapes->Foggy Cityscapes
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Cityscapes datasets/cityscapes_in_voc/ Cityscapes datasets/cityscapes_to_foggy_cityscapes/ \
  -t FoggyCityscapes datasets/foggy_cityscapes_in_voc \
  --test CityscapesTest datasets/cityscapes_in_voc/ FoggyCityscapesTest datasets/foggy_cityscapes_in_voc --finetune \
  OUTPUT_DIR logs/cyclegan/faster_rcnn_vgg_16/cityscapes2foggy


# Sim10k to Cityscapes Car
mkdir datasets/sim10k_to_cityscapes_car
cp -r datasets/sim10k/* datasets/sim10k_to_cityscapes_car
CUDA_VISIBLE_DEVICES=0 python cycle_gan.py -s Sim10k datasets/sim10k -t Cityscapes datasets/cityscapes_in_voc \
    --log logs/cyclegan/translation/sim10k2cityscapes_car --translated-source datasets/sim10k_to_cityscapes_car --image-base 256

# ResNet101 Based Faster RCNN: Sim10k -> Cityscapes Car
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s Sim10kCar datasets/sim10k Sim10kCar datasets/sim10k_to_cityscapes_car -t CityscapesCar datasets/cityscapes_in_voc/ \
  --test CityscapesCarTest datasets/cityscapes_in_voc/ --finetune \
  OUTPUT_DIR logs/cyclegan/faster_rcnn_R_101_C4/sim10k2cityscapes_car MODEL.ROI_HEADS.NUM_CLASSES 1

# VGG16 Based Faster RCNN: Sim10k -> Cityscapes Car
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_vgg_16_cityscapes.yaml \
  -s Sim10kCar datasets/sim10k Sim10kCar datasets/sim10k_to_cityscapes_car -t CityscapesCar datasets/cityscapes_in_voc/  \
  --test CityscapesCarTest datasets/cityscapes_in_voc/ --finetune \
  OUTPUT_DIR logs/cyclegan/faster_rcnn_vgg_16/sim10k2cityscapes_car MODEL.ROI_HEADS.NUM_CLASSES 1

# GTA5 to Cityscapes
mkdir datasets/gta5_to_cityscapes
cp -r datasets/synscapes_detection/* datasets/gta5_to_cityscapes
CUDA_VISIBLE_DEVICES=0 python cycle_gan.py -s GTA5 datasets/synscapes_detection -t Cityscapes datasets/cityscapes_in_voc \
    --log logs/cyclegan/translation/gta52cityscapes --translated-source datasets/gta5_to_cityscapes --image-base 256

# ResNet101 Based Faster RCNN: GTA5 -> Cityscapes
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_cityscapes.yaml \
  -s GTA5 datasets/synscapes_detection GTA5 datasets/gta5_to_cityscapes -t Cityscapes datasets/cityscapes_in_voc \
  --test CityscapesTest datasets/cityscapes_in_voc/ --finetune \
  OUTPUT_DIR logs/cyclegan/faster_rcnn_R_101_C4/gta52cityscapes
