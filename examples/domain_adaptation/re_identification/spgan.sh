# Market1501 -> Duke
# step1: train SPGAN
CUDA_VISIBLE_DEVICES=0 python spgan.py data -s Market1501 -t DukeMTMC \
--log logs/spgan/Market2Duke --translated-root data/spganM2D --seed 0
# step2: train baseline on translated source dataset
CUDA_VISIBLE_DEVICES=0 python baseline.py data/spganM2D data -s Market1501 -t DukeMTMC -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/spgan/Market2Duke

# Duke -> Market1501
# step1: train SPGAN
CUDA_VISIBLE_DEVICES=0 python spgan.py data -s DukeMTMC -t Market1501 \
--log logs/spgan/Duke2Market --translated-root data/spganD2M --seed 0
# step2: train baseline on translated source dataset
CUDA_VISIBLE_DEVICES=0 python baseline.py data/spganD2M data -s DukeMTMC -t Market1501 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/spgan/Duke2Market

# Market1501 -> MSMT17
# step1: train SPGAN
CUDA_VISIBLE_DEVICES=0 python spgan.py data -s Market1501 -t MSMT17 \
--log logs/spgan/Market2MSMT --translated-root data/spganM2S --seed 0
# step2: train baseline on translated source dataset
CUDA_VISIBLE_DEVICES=0 python baseline.py data/spganM2S data -s Market1501 -t MSMT17 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/spgan/Market2MSMT

# MSMT -> Market1501
# step1: train SPGAN
CUDA_VISIBLE_DEVICES=0 python spgan.py data -s MSMT17 -t Market1501 \
--log logs/spgan/MSMT2Market --translated-root data/spganS2M --seed 0
# step2: train baseline on translated source dataset
CUDA_VISIBLE_DEVICES=0 python baseline.py data/spganS2M data -s MSMT17 -t Market1501 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/spgan/MSMT2Market

# Duke -> MSMT
# step1: train SPGAN
CUDA_VISIBLE_DEVICES=0 python spgan.py data -s DukeMTMC -t MSMT17 \
--log logs/spgan/Duke2MSMT --translated-root data/spganD2S --seed 0
# step2: train baseline on translated source dataset
CUDA_VISIBLE_DEVICES=0 python baseline.py data/spganD2S data -s DukeMTMC -t MSMT17 -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/spgan/Duke2MSMT

# MSMT -> Duke
# step1: train SPGAN
CUDA_VISIBLE_DEVICES=0 python spgan.py data -s MSMT17 -t DukeMTMC \
--log logs/spgan/MSMT2Duke --translated-root data/spganS2D --seed 0
# step2: train baseline on translated source dataset
CUDA_VISIBLE_DEVICES=0 python baseline.py data/spganS2D data -s MSMT17 -t DukeMTMC -a reid_resnet50 \
--iters-per-epoch 800 --print-freq 80 --finetune --seed 0 --log logs/spgan/MSMT2Duke
