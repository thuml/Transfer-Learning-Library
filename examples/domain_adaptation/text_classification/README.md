# Unsupervised Domain Adaptation for Text Classification

## Installation
Currently, our model is based on word2vec supported by torchtext. 
You should first install ``torchtext==0.10.0`` before using these scripts.

```
pip install torchtext==0.10.0
```

## Dataset
Four emotion classification dataset is provided in ``data`` directory.

- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- [SemEval-2018](https://alt.qcri.org/semeval2018/index.php?id=tasks)
- [ISEAR](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)
- [Emotion-stimulus](https://metatext.io/datasets/emotion-stimulu)

We have reorganized the datasets into json files, where each line is in the format of "label,text".

## Supported Methods

Supported methods include:

- [Domain Adversarial Neural Network (DANN, only serve as a baseline)](https://arxiv.org/abs/1505.07818)
- [Margin Disparity Discrepancy (MDD)](https://arxiv.org/abs/1904.05801)
- [Localized Disparity Discrepancy (LDD)](https://arxiv.org/abs/1904.05801)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/image_classification.rst) with specified hyper-parameters.
For example, if you want to train LDD on Goemtions -> SemEval-2018, use the following script

```shell script
# Train LDD on Office-31 Goemtions -> SemEval-2018 task using word2vec.
CUDA_VISIBLE_DEVICES=0 python ldd.py -s goemtions.csv -t SemEval-2018.csv \
  --labels joy surprise sadness anger disgust fear love optimism --feature-dim 2048 --seed 0 --log logs/ldd/emtion_g2s
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.

## TODO
Support methods: TBD

## Citation
If you use these methods in your research, please consider citing.

```
@inproceedings{MDD,
    title={Bridging theory and algorithm for domain adaptation},
    author={Zhang, Yuchen and Liu, Tianle and Long, Mingsheng and Jordan, Michael},
    booktitle={ICML},
    year={2019},
}
```
