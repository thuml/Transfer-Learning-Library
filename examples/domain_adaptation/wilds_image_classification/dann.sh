CUDA_VISIBLE_DEVICES=0 python dann.py data/wilds -d "fmow" --aa "v0" --arch "densenet121" \
  --lr 0.1 --opt-level O1 --deterministic --vflip 0.5 --log logs/dann/fmow/lr_0_1_aa_v0_densenet121

CUDA_VISIBLE_DEVICES=0 python dann.py data/wilds -d "iwildcam" --aa "v0" --unlabeled-list "extra_unlabeled" --lr 1 --opt-level O1 \
  --deterministic --img-size 448 448 --crop-pct 1.0 --scale 1.0 1.0 --epochs 18 -b 24 24 --trade-off 0.3 -p 500 --metric "F1-macro_all" \
  --log logs/dann/iwildcam/lr_1_deterministic
