# Source Only Faster RCNN: VOC->Clipart
CUDA_VISIBLE_DEVICES=0 python visualize.py --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  --test Clipart datasets/clipart --save-path visualizations/source_only/voc2clipart \
  MODEL.WEIGHTS logs/source_only/faster_rcnn_R_101_C4/voc2clipart/model_final.pth

# Source Only Faster RCNN: VOC->WaterColor, Comic
CUDA_VISIBLE_DEVICES=0 python visualize.py --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  --test WaterColorTest datasets/watercolor ComicTest datasets/comic --save-path visualizations/source_only/voc2comic_watercolor \
  MODEL.ROI_HEADS.NUM_CLASSES 6 MODEL.WEIGHTS logs/source_only/faster_rcnn_R_101_C4/voc2watercolor_comic/model_final.pth
