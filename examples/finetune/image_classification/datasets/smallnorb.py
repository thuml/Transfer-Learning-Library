from datasets.tensorflow_datasets import TensorFlowDataset


class SmallnorbAzimuth(TensorFlowDataset):
    def __init__(self, tensorflow_dataset, info, root, data_dir, list_file, suffix='jpg',
                 label_name='label_azimuth', **kwargs):
        super(SmallnorbAzimuth, self).__init__(tensorflow_dataset, info, root, data_dir,
                                               list_file, suffix, label_name, **kwargs)


class SmallnorblElevation(TensorFlowDataset):
    def __init__(self, tensorflow_dataset, info, root, data_dir, list_file, suffix='jpg',
                 label_name='label_elevation', **kwargs):
        super(SmallnorblElevation, self).__init__(tensorflow_dataset, info, root, data_dir,
                                               list_file, suffix, label_name, **kwargs)

