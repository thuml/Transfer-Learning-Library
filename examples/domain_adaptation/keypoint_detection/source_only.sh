# Source Only
# Hands Dataset
CUDA_VISIBLE_DEVICES=0 python source_only.py data/RHD data/H3D_crop \
    -s RenderedHandPose -t Hand3DStudio --log logs/baseline/rhd2h3d --debug --seed 0
CUDA_VISIBLE_DEVICES=0 python source_only.py data/FreiHand data/RHD \
    -s FreiHand -t RenderedHandPose --log logs/baseline/freihand2rhd --debug --seed 0

# Body Dataset
CUDA_VISIBLE_DEVICES=0 python source_only.py data/surreal_processed data/Human36M \
    -s SURREAL -t Human36M --log logs/baseline/surreal2human36m --debug --seed 0 --rotation 30
CUDA_VISIBLE_DEVICES=0 python source_only.py data/surreal_processed data/lsp \
    -s SURREAL -t LSP --log logs/baseline/surreal2lsp --debug --seed 0 --rotation 30

# Oracle Results
CUDA_VISIBLE_DEVICES=0 python source_only.py data/H3D_crop data/H3D_crop \
    -s Hand3DStudio -t Hand3DStudio --log logs/oracle/h3d --debug --seed 0
CUDA_VISIBLE_DEVICES=0 python source_only.py data/Human36M data/Human36M \
    -s Human36M -t Human36M --log logs/oracle/human36m --debug --seed 0 --rotation 30
