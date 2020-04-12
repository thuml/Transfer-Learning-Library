Domain-Adaptation-Library


Installation
============

DALIB requires PyTorch 1.4 or newer.

pip:

```bash
    pip install -i https://test.pypi.org/simple/ dalib
```

    
Documentation
=============
You can find the API documentation on the website: [DALIB API](https://192.168.6.114:9000/index.html)

Also, we provide a tutorial and examples in the directory `examples`. A typical usage is 
```shell script
# Train a DANN on Office-31 Amazon->Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
python examples/dann.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 20
```

In the directory `examples`, you can find all the necessary running scripts to reproduce the benchmarks.

Contributing
============
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

Disclaimer on Datasets
======================

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
