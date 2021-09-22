#!/bin/bash
dataset=UCIHAR
tasks=(
  2 11
  7 13
  12 16
  12 18
  9 18
  14 19
  18 23
  6 23
  7 24
  17 25
)
n_tasks=`expr ${#tasks[*]} / 2`
echo $n_tasks
for ((i=0;i<$n_tasks;i++));
do
  source=${tasks[`expr 2 \* $i + 1`]}
  target=${tasks[`expr 2 \* $i + 2`]}
  echo CUDA_VISIBLE_DEVICES=3 python dann.py data -d $dataset -s $source -t $target -a fcn --epochs 5 --seed 0 --log logs/dann/${dataset}_${source}to${target}
  CUDA_VISIBLE_DEVICES=3 python dann.py data -d $dataset -s $source -t $target -a fcn --epochs 5 --seed 0 --log logs/dann/${dataset}_${source}to${target}
done


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
  echo CUDA_VISIBLE_DEVICES=1 python dann.py data -d $dataset -s $source -t $target -a fcn --epochs 10 --seed 0 --log logs/dann/${dataset}_${source}to${target}
#  CUDA_VISIBLE_DEVICES=1 python dann.py data -d $dataset -s $source -t $target -a fcn --epochs 10 --seed 0 --log logs/dann/${dataset}_${source}to${target}
done


dataset=WISDMAR
tasks=(
  1 11
  3 11
  4 15
  2 25
  25 29
  7 30
  21 31
  2 32
  1 7
  0 8
)
n_tasks=`expr ${#tasks[*]} / 2`
echo $n_tasks
for ((i=0;i<$n_tasks;i++));
do
  source=${tasks[`expr 2 \* $i + 1`]}
  target=${tasks[`expr 2 \* $i + 2`]}
  echo CUDA_VISIBLE_DEVICES=0 python dann.py data -d $dataset -s $source -t $target -a fcn --epochs 5 --seed 0 --log logs/dann/${dataset}_${source}to${target}
  CUDA_VISIBLE_DEVICES=3 python dann.py data -d $dataset -s $source -t $target -a fcn --epochs 5 --seed 0 --log logs/dann/${dataset}_${source}to${target}
done

