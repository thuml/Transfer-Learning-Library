# civilcomments
CUDA_VISIBLE_DEVICES=2 python erm.py /data/wilds -d "civilcomments" --lr 1e-05  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg"
CUDA_VISIBLE_DEVICES=2 python erm.py /data/wilds -d "civilcomments" --lr 2e-05  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg"
CUDA_VISIBLE_DEVICES=2 python erm.py /data/wilds -d "civilcomments" --lr 1e-06  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg"
CUDA_VISIBLE_DEVICES=2 python erm.py /data/wilds -d "civilcomments" --lr 2e-06  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg"

CUDA_VISIBLE_DEVICES=2 python erm.py /data/wilds -d "civilcomments" --lr 1e-05  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg" --max_token_length 300 --wd 0.01 --uniform_over_groups --groupby_fields y black
CUDA_VISIBLE_DEVICES=2 python erm.py /data/wilds -d "civilcomments" --lr 2e-05  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg" --max_token_length 300 --wd 0.01 --uniform_over_groups --groupby_fields y black
CUDA_VISIBLE_DEVICES=2 python erm.py /data/wilds -d "civilcomments" --lr 1e-06  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg" --max_token_length 300 --wd 0.01 --uniform_over_groups --groupby_fields y black
CUDA_VISIBLE_DEVICES=2 python erm.py /data/wilds -d "civilcomments" --lr 2e-06  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg" --max_token_length 300 --wd 0.01 --uniform_over_groups --groupby_fields y black

CUDA_VISIBLE_DEVICES=3 python erm.py /data/wilds -d "civilcomments" --lr 1e-05  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg" --max_token_length 300 --wd 0.01 --uniform_over_groups --groupby_fields y black --seed 0
CUDA_VISIBLE_DEVICES=3 python erm.py /data/wilds -d "civilcomments" --lr 1e-05  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg" --max_token_length 300 --wd 0.01 --uniform_over_groups --groupby_fields y black --seed 1
CUDA_VISIBLE_DEVICES=3 python erm.py /data/wilds -d "civilcomments" --lr 1e-05  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg" --max_token_length 300 --wd 0.01 --uniform_over_groups --groupby_fields y black --seed 2
CUDA_VISIBLE_DEVICES=3 python erm.py /data/wilds -d "civilcomments" --lr 1e-05  --opt-level O1 --log logs/erm/civilcomments/test --unlabeled-list "extra_unlabeled" --metric "acc_wg" --max_token_length 300 --wd 0.01 --uniform_over_groups --groupby_fields y black --seed 3

# amazon
CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 24 24 --metric "10th_percentile_acc" --lr 1e-5
CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 24 24 --metric "10th_percentile_acc" --lr 1e-6
CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 24 24 --metric "10th_percentile_acc" --lr 2e-6

CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 24 24 --metric "10th_percentile_acc" --lr 1e-5 --max_token_length 512 --wd 0.01
CUDA_VISIBLE_DEVICES=0 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 24 24 --metric "10th_percentile_acc" --lr 2e-5 --max_token_length 512 --wd 0.01
CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 24 24 --metric "10th_percentile_acc" --lr 1e-6 --max_token_length 512 --wd 0.01
CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 24 24 --metric "10th_percentile_acc" --lr 2e-6 --max_token_length 512 --wd 0.01

CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 24 24 --metric "10th_percentile_acc" --lr 1e-5 --max_token_length 512 --wd 0.01 --seed 1
CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 24 24 --metric "10th_percentile_acc" --lr 1e-5 --max_token_length 512 --wd 0.01 --seed 2

CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 8 8 --metric "10th_percentile_acc" --lr 1e-5 --max_token_length 512 --wd 0.01 --seed 0
CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 8 8 --metric "10th_percentile_acc" --lr 1e-5 --max_token_length 512 --wd 0.01 --seed 1
CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test -b 8 8 --metric "10th_percentile_acc" --lr 1e-5 --max_token_length 512 --wd 0.01 --seed 2

CUDA_VISIBLE_DEVICES=1 python erm.py /data/wilds -d "amazon" --opt-level O1 --log logs/erm/amazon/test1 -b 8 8 --metric "10th_percentile_acc" --lr 1e-5 --max_token_length 512 --wd 0.01 --seed 0
