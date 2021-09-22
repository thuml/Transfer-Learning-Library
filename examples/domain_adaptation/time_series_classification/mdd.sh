#!/bin/bash
dataset=UCIHHAR
tasks=(
  1 3
  3 5
  4 5
  0 6
  1 6
  4 6
  5 6
  2 7
  3 8
  5 8
)
n_tasks=`expr ${#tasks[*]} / 2`
echo $n_tasks
for ((i=0;i<$n_tasks;i++));
do
  source=${tasks[`expr 2 \* $i + 1`]}
  target=${tasks[`expr 2 \* $i + 2`]}
  echo CUDA_VISIBLE_DEVICES=1 python mdd.py data -d $dataset -s $source -t $target -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/${dataset}_${source}to${target}
  CUDA_VISIBLE_DEVICES=1 python mdd.py data -d $dataset -s $source -t $target -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/${dataset}_${source}to${target}
done

CUDA_VISIBLE_DEVICES=1 python mdd.py data -d UCIHHAR -s 1 -t 3 -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/UCIHHAR_1to3
CUDA_VISIBLE_DEVICES=1 python mdd.py data -d UCIHHAR -s 3 -t 5 -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/UCIHHAR_3to5
CUDA_VISIBLE_DEVICES=1 python mdd.py data -d UCIHHAR -s 4 -t 5 -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/UCIHHAR_4to5
CUDA_VISIBLE_DEVICES=1 python mdd.py data -d UCIHHAR -s 0 -t 6 -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/UCIHHAR_0to6
CUDA_VISIBLE_DEVICES=1 python mdd.py data -d UCIHHAR -s 1 -t 6 -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/UCIHHAR_1to6
CUDA_VISIBLE_DEVICES=1 python mdd.py data -d UCIHHAR -s 4 -t 6 -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/UCIHHAR_4to6
CUDA_VISIBLE_DEVICES=1 python mdd.py data -d UCIHHAR -s 5 -t 6 -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/UCIHHAR_5to6
CUDA_VISIBLE_DEVICES=1 python mdd.py data -d UCIHHAR -s 2 -t 7 -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/UCIHHAR_2to7
CUDA_VISIBLE_DEVICES=1 python mdd.py data -d UCIHHAR -s 3 -t 8 -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/UCIHHAR_3to8
CUDA_VISIBLE_DEVICES=1 python mdd.py data -d UCIHHAR -s 5 -t 8 -a fcn --epochs 20 --seed 0 --margin 4. --log logs/mdd/UCIHHAR_5to8