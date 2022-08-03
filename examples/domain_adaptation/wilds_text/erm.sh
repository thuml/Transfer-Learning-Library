# civilcomments
CUDA_VISIBLE_DEVICES=0 python erm.py data/wilds -d "civilcomments" --unlabeled-list "extra_unlabeled" \
  --uniform-over-groups --groupby-fields y black --max-token-length 300 --lr 1e-05 --metric "acc_wg" \
  --seed 0 --deterministic --log logs/erm/civilcomments

# amazon
CUDA_VISIBLE_DEVICES=0 python erm.py data/wilds -d "amazon" --max-token-length 512 \
  --lr 1e-5 -b 24 24 --epochs 3 --metric "10th_percentile_acc" --seed 0 --deterministic --log logs/erm/amazon
