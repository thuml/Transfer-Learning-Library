CUDA_VISIBLE_DEVICES=3 python mdd.py -s goemtions.csv -t SemEval-2018.csv \
  --labels joy surprise sadness anger disgust fear love optimism --feature-dim 2048 --seed 0 --log logs/mdd/emtion_g2s
CUDA_VISIBLE_DEVICES=3 python mdd.py -s goemtions.csv -t isear.csv \
  --labels joy surprise sadness anger disgust fear --feature-dim 1024 --seed 0 --log logs/mdd/emtion_g2i
CUDA_VISIBLE_DEVICES=3 python mdd.py -s goemtions.csv -t emotion-stimulus.csv \
  --labels joy sadness anger disgust fear --feature-dim 1024 --seed 0 --log logs/mdd/emtion_g2e
CUDA_VISIBLE_DEVICES=3 python mdd.py -s SemEval-2018.csv -t isear.csv \
  --labels joy surprise sadness anger disgust fear --feature-dim 512 --seed 0 --log logs/mdd/emtion_s2i
CUDA_VISIBLE_DEVICES=3 python mdd.py -s SemEval-2018.csv -t emotion-stimulus.csv  \
  --labels joy sadness anger fear disgust surprise --feature-dim 512 --seed 0 --log logs/mdd/emtion_s2e
CUDA_VISIBLE_DEVICES=3 python mdd.py -s isear.csv -t emotion-stimulus.csv \
  --labels joy sadness anger disgust fear --feature-dim 1024 --seed 0 --log logs/mdd/emtion_i2e


