# Hands Dataset
CUDA_VISIBLE_DEVICES=0 python regda.py data/RHD data/H3D_crop \
    -s RenderedHandPose -t Hand3DStudio --seed 0 --debug --log logs/regda/rhd2h3d
CUDA_VISIBLE_DEVICES=0 python regda.py data/FreiHand data/RHD \
    -s FreiHand -t RenderedHandPose --seed 0 --debug --log logs/regda/freihand2rhd

# Body Dataset
CUDA_VISIBLE_DEVICES=0 python regda.py data/surreal_processed data/Human36M \
    -s SURREAL -t Human36M --finetune --seed 0 --debug --rotation 30 --epochs 10 --log logs/regda/surreal2human36m
CUDA_VISIBLE_DEVICES=0 python regda.py data/surreal_processed data/lsp \
    -s SURREAL -t LSP --finetune --seed 0 --debug --rotation 30 --log logs/regda/surreal2lsp
