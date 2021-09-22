CUDA_VISIBLE_DEVICES=2 python dann.py -s goemtions.csv -t SemEval-2018.csv \
  --labels joy surprise sadness anger disgust fear love optimism --seed 0 --log logs/dann/emtion_g2s
CUDA_VISIBLE_DEVICES=2 python dann.py -s goemtions.csv -t isear.csv \
  --labels joy surprise sadness anger disgust fear --seed 0 --log logs/dann/emtion_g2i
CUDA_VISIBLE_DEVICES=2 python dann.py -s goemtions.csv -t emotion-stimulus.csv \
  --labels joy sadness anger disgust fear --seed 0 --log logs/dann/emtion_g2e
CUDA_VISIBLE_DEVICES=2 python dann.py -s SemEval-2018.csv -t isear.csv \
  --labels joy surprise sadness anger disgust fear --seed 0 --log logs/dann/emtion_s2i
CUDA_VISIBLE_DEVICES=2 python dann.py -s SemEval-2018.csv -t emotion-stimulus.csv  \
  --labels joy sadness anger fear disgust surprise --seed 0 --log logs/dann/emtion_s2e
CUDA_VISIBLE_DEVICES=2 python dann.py -s isear.csv -t emotion-stimulus.csv \
  --labels joy sadness anger disgust fear --seed 0 --log logs/dann/emtion_i2e
