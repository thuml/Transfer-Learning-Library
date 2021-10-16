# regda_fast is provided by https://github.com/YouJiacheng?tab=repositories
# On single V100(16G), overall adversarial training time is reduced by about 40%.
# yet the PCK might drop 1% for each dataset.
# Hands Dataset
CUDA_VISIBLE_DEVICES=0 python regda_fast.py data/RHD data/H3D_crop \
    -s RenderedHandPose -t Hand3DStudio --seed 0 --debug --log logs/regda_fast/rhd2h3d
CUDA_VISIBLE_DEVICES=0 python regda_fast.py data/FreiHand data/RHD \
    -s FreiHand -t RenderedHandPose --seed 0 --debug --log logs/regda_fast/freihand2rhd

# Body Dataset
CUDA_VISIBLE_DEVICES=0 python regda_fast.py data/surreal_processed data/Human36M \
    -s SURREAL -t Human36M --seed 0 --debug --rotation 30 --epochs 10 --log logs/regda_fast/surreal2human36m
CUDA_VISIBLE_DEVICES=0 python regda_fast.py data/surreal_processed data/lsp \
    -s SURREAL -t LSP --seed 0 --debug --rotation 30 --log logs/regda_fast/surreal2lsp
