# Unsupervised Domain Adaptation for Object Detection


## Installation
Our code is based on [Detectron latest(v0.6)](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), please install it before usage.


## Dataset

You need to prepare following datasets manually if you want to use them:
- [Cityscapes](https://www.cityscapes-dataset.com/)
- VOC0712
- Clipart
- WaterColor
- Comic
- Sim10k


## Supported Methods

Supported methods include:

- [Cycle-Consistent Adversarial Networks (CycleGAN)](https://arxiv.org/pdf/1703.10593.pdf)
- [Decoupled Adaptation for Cross-Domain Object Detection (D-adapt)](https://arxiv.org/abs/2110.02578)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/object_detection.rst) with specified hyper-parameters.
The basic training pipeline is as follows.

#### Prepare Datasets
Our code is based on Detectron due to its performance and scalability. Detectron expects dataset to be either in `VOC` format
or `CoCo` format. VOC0712, Clipart, WaterColor and Comic datasets are already in `VOC` format. For other datasets and perhaps your own
datasets, you need to prepare them manually.

Below we use Cityscapes dataset as an example. First download the raw [dataset](https://www.cityscapes-dataset.com/), 
and unzip them under the directory like
```
object_detction/datasets/cityscape
├── gtFine
├── leftImg8bit
├── leftImg8bit_foggy
└── ...
```
Then run 
```
python prepare_cityscapes_to_voc.py 
```
This will automatically generate dataset in `VOC` format.
```
object_detction/datasets/cityscapes_in_voc
├── Annotations
├── ImageSets
├── JPEGImages
```

**Note**: you may need to install `pascal_voc_writer` via `pip install pascal_voc_writer` when converting the dataset format. To use your own datasets, you 
should convert them into corresponding format. See `prepare_gta5_to_voc.py` and `prepare_cityscapes_to_voc.py` for some help.

#### Run our code
The following command trains a Faster-RCNN detector on task VOC->Clipart, with only source (VOC) data.
```
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 -t Clipart datasets/clipart \
  --test VOC2007Test datasets/VOC2007 Clipart datasets/clipart --finetune \
  OUTPUT_DIR logs/source_only/faster_rcnn_R_101_C4/voc2clipart
```
Explanation of some arguments
- `--config-file`: path to config file that specifies training hyper-parameters.
- `-s`: a list that specifies source datasets, for each dataset you should pass in a `(name, path)` pair, in the
    above command, there are two source datasets **VOC2007** and **VOC2012**.
- `-t`: a list that specifies target datasets, same format as above.
- `--test`: a list that specifiers test datasets, same format as above.

### VOC->Clipart

|                         |          | AP   | AP50 | AP75 | aeroplane | bicycle | bird | boat | bottle | bus  | car  | cat  | chair | cow  | diningtable | dog  | horse | motorbike | person | pottedplant | sheep | sofa | train | tvmonitor |
|-------------------------|----------|------|------|------|-----------|---------|------|------|--------|------|------|------|-------|------|-------------|------|-------|-----------|--------|-------------|-------|------|-------|-----------|
| Faster RCNN (ResNet101) | Source   | 14.9 | 29.3 | 12.6 | 29.6      | 38.0    | 24.7 | 21.7 | 31.9   | 48.0 | 30.8 | 15.9 | 32.0  | 19.2 | 18.2        | 12.1 | 28.2  | 48.8      | 38.3   | 34.6        | 3.8   | 22.5 | 43.7  | 44.0      |
|                         | CycleGAN | 20.0 | 37.7 | 18.3 | 37.1      | 41.9    | 29.9 | 26.5 | 40.9   | 65.1 | 37.8 | 23.8 | 40.7  | 48.9 | 12.7        | 14.4 | 27.8  | 63.0      | 55.1   | 40.1        | 8.0   | 30.7 | 54.1  | 55.7      |
|                         | D-adapt  | 24.8 | 49.0 | 21.5 | 56.4      | 63.2    | 42.3 | 40.9 | 45.3   | 77.0 | 48.7 | 25.4 | 44.3  | 58.4 | 31.4        | 24.5 | 47.1  | 75.3      | 69.3   | 43.5        | 27.9  | 34.1 | 60.7  | 64.0      |
|                         |          |      |      |      |           |         |      |      |        |      |      |      |       |      |             |      |       |           |        |             |       |      |       |           |
| RetinaNet               | Source   | 18.3 | 32.2 | 17.6 | 34.2      | 42.4    | 27.0 | 21.6 | 36.8   | 48.4 | 35.9 | 16.4 | 38.9  | 22.6 | 27.0        | 15.1 | 27.1  | 46.7      | 42.1   | 36.2        | 8.3   | 29.5 | 42.1  | 46.2      |
|                         | D-adapt  | 25.1 | 46.3 | 23.9 | 47.4      | 65.0    | 33.1 | 37.5 | 56.8   | 61.2 | 55.1 | 27.3 | 45.5  | 51.8 | 29.1        | 29.6 | 38.0  | 74.5      | 66.7   | 46.0        | 24.2  | 29.3 | 54.2  | 53.8      |

### VOC->WaterColor

|                         | AP   | AP50 | AP75 | bicycle | bird | car  | cat  | dog  | person |
|-------------------------|------|------|------|---------|------|------|------|------|--------|
| Faster RCNN (ResNet101) | 23.0 | 45.9 | 18.5 | 71.1    | 48.3 | 48.6 | 23.7 | 23.3 | 60.3   |
| CycleGAN                | 24.9 | 50.8 | 22.4 | 75.8    | 52.1 | 49.8 | 30.1 | 33.4 | 63.6   |
| D-adapt                 | 28.5 | 57.5 | 23.6 | 77.4    | 54.0 | 52.8 | 43.9 | 48.1 | 68.9   |
| Target                  | 23.8 | 51.3 | 17.4 | 48.5    | 54.7 | 41.3 | 36.2 | 52.6 | 74.6   |

### VOC->Comic

|                         |  AP  | AP50 | AP75 | bicycle | bird |  car |  cat |  dog | person |
|:-----------------------:|:----:|:----:|:----:|:-------:|:----:|:----:|:----:|:----:|:------:|
| Faster RCNN (ResNet101) | 13.0 | 25.5 | 11.4 |   33.0  | 15.8 | 28.9 | 16.8 | 19.6 |  39.0  |
|         CycleGAN        | 16.9 | 34.6 | 14.2 |   28.1  | 25.7 | 37.7 | 28.0 | 33.8 |  54.1  |
|         D-adapt         | 20.8 | 41.1 | 18.5 |   49.4  | 25.7 | 43.3 | 36.9 | 32.7 |  58.5  |
|          Target         | 21.9 | 44.6 | 16.0 |   40.7  | 32.3 | 38.3 | 43.9 | 41.3 |  71.0  |


### Cityscapes->Foggy Cityscapes
|                         |          |  AP  | AP50 | AP75 | bicycle |  bus |  car | motorcycle | person | rider | train | truck |
|:-----------------------:|:--------:|:----:|:----:|:----:|:-------:|:----:|:----:|:----------:|:------:|:-----:|:-----:|:-----:|
|   Faster RCNN (VGG16)   |  Source  | 14.3 | 25.9 | 13.2 |   33.6  | 27.0 | 40.0 |    22.3    |  31.3  |  38.5 |  2.3  |  12.2 |
|                         | CycleGAN | 22.5 | 41.6 | 20.7 |   46.5  | 41.5 | 62.0 |    33.8    |  45.0  |  54.5 |  21.7 |  27.7 |
|                         |  D-adapt | 19.4 | 38.1 | 17.5 |   42.0  | 36.8 | 58.1 |    32.2    |  43.1  |  51.8 |  14.6 |  26.3 |
|                         |  Target  | 24.0 | 45.3 | 21.3 |   45.9  | 47.4 | 67.3 |    39.7    |  49.0  |  53.2 |  30.0 |  29.6 |
|                         |          |      |      |      |         |      |      |            |        |       |       |       |
| Faster RCNN (ResNet101) |  Source  | 18.8 | 33.3 | 19.0 |   36.1  | 34.5 | 43.8 |    24.0    |  36.3  |  39.9 |  29.1 |  22.8 |
|                         | CycleGAN | 22.9 | 41.8 | 21.9 |   42.0  | 44.5 | 57.6 |    36.3    |  40.9  |  48.0 |  30.8 |  34.3 |
|                         |  D-adapt | 22.7 | 42.4 | 21.6 |   41.8  | 44.4 | 56.6 |    31.4    |  41.8  |  48.6 |  42.3 |  32.4 |
|                         |  Target  | 25.5 | 45.3 | 24.3 |   41.9  | 53.2 | 63.4 |    36.1    |  42.6  |  47.9 |  42.4 |  35.3 |

### Sim10k->Cityscapes Car

|                         |          |  AP  | AP50 | AP75 |
|:-----------------------:|:--------:|:----:|:----:|:----:|
|   Faster RCNN (VGG16)   |  Source  | 24.8 | 43.4 | 23.6 |
|                         | CycleGAN | 29.3 | 51.9 | 28.6 |
|                         |  D-adapt | 23.6 | 48.5 | 18.7 |
|                         |  Target  | 24.8 | 43.4 | 23.6 |
|                         |          |      |      |      |
| Faster RCNN (ResNet101) |  Source  | 24.6 | 44.4 | 23.0 |
|                         | CycleGAN | 26.5 | 47.4 | 24.0 |
|                         |  D-adapt | 27.4 | 51.9 | 25.7 |
|                         |  Target  | 24.6 | 44.4 | 23.0 |

### Visualization

## TODO
Support methods: SWDA, Global/Local Alignment, Dataset Installation Instruction, Tutorial, visualization code


## Citation
If you use these methods in your research, please consider citing.

```
@misc{jiang2021decoupled,
      title={Decoupled Adaptation for Cross-Domain Object Detection}, 
      author={Junguang Jiang and Baixu Chen and Jianmin Wang and Mingsheng Long},
      year={2021},
      eprint={2110.02578},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{CycleGAN,
    title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
    author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
    booktitle={ICCV},
    year={2017}
}
```
