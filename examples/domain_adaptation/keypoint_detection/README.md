# Unsupervised Domain Adaptation for Keypoint Detection

## Dataset
Following datasets can be downloaded automatically:

- Rendered Handpose Dataset
- Hand-3d-Studio Dataset
- FreiHAND Dataset
- Surreal Dataset
- LSP Dataset

You need to prepare following datasets manually if you want to use them:
- [Human3.6M Dataset](http://vision.imar.ro/human3.6m/description.php)

and prepare them following [Documentations for Human3.6M Dataset](/common/vision/datasets/keypoint_detection/human36m.py).

## Supported Methods

Supported methods include:

- Regressive Domain Adaptation (RegDA)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/keypoint_detection_da.rst) with specified hyper-parameters.
For example, if you want to train DANN on Office31, use the following script

```shell script
# Train a RegDA on RHD -> H3D task using PoseResNet.
# Assume you have put the datasets under the path `data/RHD` and  `data/H3D_crop`, 
# or you are glad to download the datasets automatically from the Internet to this path
CUDA_VISIBLE_DEVICES=0 python regda.py data/RHD data/H3D_crop \
    -s RenderedHandPose -t Hand3DStudio --finetune --seed 0 --debug --log logs/regda/rhd2h3d
```

For more information please refer to [Get Started](/docs/get_started/quickstart.rst) for help.

## TODO
Support methods:  CycleGAN, CyCADA

## Citation
If you use these methods in your research, please consider citing.

```
@InProceedings{RegDA,
  author    = {Junguang Jiang and
               Yifei Ji and
               Ximei Wang and
               Yufeng Liu and
               Jianmin Wang and
               Mingsheng Long},
  title     = {Regressive Domain Adaptation for Unsupervised Keypoint Detection},
  booktitle = {CVPR},
  year = {2021}
}

```
