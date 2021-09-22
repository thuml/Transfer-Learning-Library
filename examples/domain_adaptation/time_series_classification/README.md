# Unsupervised Domain Adaptation for Time-Series Classification

## Dataset
3 time-series classification dataset is provided,

- UCIHAR
- UCIHHAR
- WISDM_AR

Following scripts will download the dataset and preprocess them in directory ``data``.

```shell
cd preprocess
python prepare_datasets.py
```

The data preparation code is modified from [https://github.com/floft/codats](https://github.com/floft/codats).

## Supported Methods

Supported methods include:

- Domain Adversarial Neural Network (DANN)
- Deep Adaptation Network (DAN)
- Margin Disparity Discrepancy (MDD)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/unsupervised_da.rst) with specified hyper-parameters.
For example, if you want to train DANN on UCIHHAR, use the following script

```shell script
CUDA_VISIBLE_DEVICES=1 python dann.py data -d UCIHHAR -s 1 -t 3 -a fcn --epochs 10 --seed 0 --log logs/dann/UCIHHAR_1to3
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.


## Citation
If you use these methods in your research, please consider citing.

```
@inproceedings{DANN,
	Author = {Ganin, Yaroslav and Lempitsky, Victor},
	Booktitle = {ICML},
	Title = {Unsupervised domain adaptation by backpropagation},
	Year = {2015}
}

@inproceedings{DAN,
	author    = {Mingsheng Long and
	Yue Cao and
	Jianmin Wang and
	Michael I. Jordan},
	title     = {Learning Transferable Features with Deep Adaptation Networks},
	booktitle = {ICML},
	year      = {2015},
}

@inproceedings{MDD,
    title={Bridging theory and algorithm for domain adaptation},
    author={Zhang, Yuchen and Liu, Tianle and Long, Mingsheng and Jordan, Michael},
    booktitle={ICML},
    year={2019},
}

```
