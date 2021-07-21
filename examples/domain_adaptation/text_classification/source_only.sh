#!/usr/bin/env bash
# GLUE 2-classification (entailment or non-entailment)
# Oracle Results (train on target)
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s mnli -t mnli --seed 0 --log logs/oracle/GLUE_mnli
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qnli -t qnli --seed 0 --log logs/oracle/GLUE_qnli
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qqp -t qqp --seed 0 --log logs/oracle/GLUE_qqp
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s snli -t snli --seed 0 --log logs/oracle/GLUE_snli
# Source Only
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s mnli -t qnli --phase test --load_from logs/oracle/GLUE_mnli/checkpoints/best.pth --log logs/src_only/GLUE_mnli2qnli
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s mnli -t qqp --phase test --load_from logs/oracle/GLUE_mnli/checkpoints/best.pth --log logs/src_only/GLUE_mnli2qqp
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s mnli -t snli --phase test --load_from logs/oracle/GLUE_mnli/checkpoints/best.pth --log logs/src_only/GLUE_mnli2snli
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qnli -t mnli --phase test --load_from logs/oracle/GLUE_qnli/checkpoints/best.pth --log logs/src_only/GLUE_qnli2mnli
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qnli -t qqp --phase test --load_from logs/oracle/GLUE_qnli/checkpoints/best.pth --log logs/src_only/GLUE_qnli2qqp
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qnli -t snli --phase test --load_from logs/oracle/GLUE_qnli/checkpoints/best.pth --log logs/src_only/GLUE_qnli2snli
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qqp -t mnli --phase test --load_from logs/oracle/GLUE_qqp/checkpoints/best.pth --log logs/src_only/GLUE_qqp2mnli
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qqp -t qnli --phase test --load_from logs/oracle/GLUE_qqp/checkpoints/best.pth --log logs/src_only/GLUE_qqp2qnli
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qqp -t snli --phase test --load_from logs/oracle/GLUE_qqp/checkpoints/best.pth --log logs/src_only/GLUE_qqp2snli
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s snli -t mnli --phase test --load_from logs/oracle/GLUE_snli/checkpoints/best.pth --log logs/src_only/GLUE_snli2mnli
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s snli -t qnli --phase test --load_from logs/oracle/GLUE_snli/checkpoints/best.pth --log logs/src_only/GLUE_snli2qnli
#CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s snli -t qqp --phase test --load_from logs/oracle/GLUE_snli/checkpoints/best.pth --log logs/src_only/GLUE_snli2qqp
#

CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s mnli -t qnli --log logs/src_only/GLUE_mnli2qnli
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s mnli -t qqp --log logs/src_only/GLUE_mnli2qqp
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s mnli -t snli --log logs/src_only/GLUE_mnli2snli
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qnli -t mnli --log logs/src_only/GLUE_qnli2mnli
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qnli -t qqp --log logs/src_only/GLUE_qnli2qqp
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qnli -t snli --log logs/src_only/GLUE_qnli2snli
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qqp -t mnli --log logs/src_only/GLUE_qqp2mnli
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qqp -t qnli --log logs/src_only/GLUE_qqp2qnli
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s qqp -t snli --log logs/src_only/GLUE_qqp2snli
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s snli -t mnli --log logs/src_only/GLUE_snli2mnli
CUDA_VISIBLE_DEVICES=5 python source_only.py -d GLUE -s snli -t qnli --log logs/src_only/GLUE_snli2qnli
CUDA_VISIBLE_DEVICES=6 python source_only.py -d GLUE -s snli -t qqp --log logs/src_only/GLUE_snli2qqp


#CUDA_VISIBLE_DEVICES=7 python source_only.py --root data/wilds/ -d amazon --log logs/src_only/amazon2 --max_length 512 -b 8 --lr 1e-5
#CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d civilcomments --log logs/src_only/civilcomments --max_length 300 -b 16 --lr 1e-5
#
#CUDA_VISIBLE_DEVICES=3 python source_only.py --root data/wilds/ -d amazon --arch distilbert-base-uncased --log logs/src_only_distilbert/amazon2 --max_length 512 -b 8 --lr 1e-5
#CUDA_VISIBLE_DEVICES=3 python source_only.py --root data/wilds/ -d civilcomments --arch distilbert-base-uncased --log logs/src_only_distilbert/civilcomments --max_length 300 -b 16 --lr 1e-5

# Amazon
domains=(books movies_and_tv electronics home_and_kitchen)
for source in ${domains[@]}
  do
  for target in ${domains[@]}
  do
    if [ "$source" != "$target" ];then
    echo CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s $source -t $target --log logs/src_only/amazon_${source}_to_${target}
    fi
  done
done

CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s books -t movies_and_tv --log logs/src_only/amazon_books_to_movies_and_tv
CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s books -t electronics --log logs/src_only/amazon_books_to_electronics
CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s books -t home_and_kitchen --log logs/src_only/amazon_books_to_home_and_kitchen
CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s movies_and_tv -t books --log logs/src_only/amazon_movies_and_tv_to_books
CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s movies_and_tv -t electronics --log logs/src_only/amazon_movies_and_tv_to_electronics
CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s movies_and_tv -t home_and_kitchen --log logs/src_only/amazon_movies_and_tv_to_home_and_kitchen
CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s electronics -t books --log logs/src_only/amazon_electronics_to_books
CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s electronics -t movies_and_tv --log logs/src_only/amazon_electronics_to_movies_and_tv
CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s electronics -t home_and_kitchen --log logs/src_only/amazon_electronics_to_home_and_kitchen
CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s home_and_kitchen -t books --log logs/src_only/amazon_home_and_kitchen_to_books
CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s home_and_kitchen -t movies_and_tv --log logs/src_only/amazon_home_and_kitchen_to_movies_and_tv
CUDA_VISIBLE_DEVICES=4 python source_only.py --root data/wilds/ -d amazon -s home_and_kitchen -t electronics --log logs/src_only/amazon_home_and_kitchen_to_electronics