"""
Datasets

Load the desired datasets into memory so we can write them to tfrecord files
in generate_tfrecords.py
"""
import os
import re
import io
import zipfile
import tarfile
import rarfile  # pip install rarfile
import scipy.io
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from absl import flags

from normalization import calc_normalization_jagged, apply_normalization_jagged

FLAGS = flags.FLAGS

flags.DEFINE_enum("normalize", "meanstd", ["none", "minmax", "meanstd"], "How to normalize data")

list_of_datasets = {}


def register_dataset(name):
    """ Add dataset to the list of datsets, e.g. add @register_dataset("name")
    before a class definition """
    assert name not in list_of_datasets, "duplicate dataset named " + name

    def decorator(cls):
        list_of_datasets[name] = cls
        return cls

    return decorator


def get_dataset(name):
    """ Based on the given name, get the correct dataset processor """
    assert name in list_of_datasets.keys(), \
        "Unknown dataset name " + name
    return list_of_datasets[name]


def get_dataset_users(name):
    """ Get list of users for a dataset """
    return get_dataset(name).users


def call_dataset(name, *args, **kwargs):
    """ Based on the given name, call the correct dataset processor """
    return get_dataset(name)(*args, **kwargs)


def list_datasets():
    """ Returns list of all the available datasets """
    return list(list_of_datasets.keys())


def zero_to_n(n):
    """ Return [0, 1, 2, ..., n] """
    return list(range(0, n+1))


def one_to_n(n):
    """ Return [1, 2, 3, ..., n] """
    return list(range(1, n+1))


class Dataset:
    """
    Base class for datasets

    class Something(Dataset):
        num_classes = 2
        class_labels = ["class1", "class2"]
        window_size = 250
        window_overlap = False

        def __init__(self, *args, **kwargs):
            super().__init__(Something.num_classes, Something.class_labels,
                Something.window_size, Something.window_overlap,
                *args, **kwargs)

        def process(self, data, labels):
            ...
            return super().process(data, labels)

        def load(self):
            ...
            return train_data, train_labels, test_data, test_labels

    Also, add to the datasets={"something": Something, ...} dictionary below.
    """
    already_normalized = False

    def __init__(self, num_classes, class_labels, window_size, window_overlap,
            feature_names=None, test_percent=0.2):
        """
        Initialize dataset

        Must specify num_classes and class_labels (the names of the classes).

        For example,
            Dataset(num_classes=2, class_labels=["class1", "class2"])

        This calls load() to get the data, process() to normalize, convert to
        float, etc.

        At the end, look at dataset.{train,test}_{data,labels}
        """
        # Sanity checks
        assert num_classes == len(class_labels), \
            "num_classes != len(class_labels)"

        # Set parameters
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.feature_names = feature_names
        self.test_percent = test_percent

        # Load the dataset
        train_data, train_labels, test_data, test_labels = self.load()

        if train_data is not None and train_labels is not None:
            self.train_data, self.train_labels = \
                self.process(train_data, train_labels)
        else:
            self.train_data = None
            self.train_labels = None

        if test_data is not None and test_labels is not None:
            self.test_data, self.test_labels = \
                self.process(test_data, test_labels)
        else:
            self.test_data = None
            self.test_labels = None

    def load(self):
        raise NotImplementedError("must implement load() for Dataset class")

    def download_dataset(self, files_to_download, url):
        """
        Download url/file for file in files_to_download
        Returns: the downloaded filenames for each of the files given
        """
        downloaded_files = []

        for f in files_to_download:
            downloaded_files.append(tf.keras.utils.get_file(
                fname=f, origin=url+"/"+f))

        return downloaded_files

    def process(self, data, labels):
        """ Perform conversions, etc. If you override,
        you should `return super().process(data, labels)` to make sure these
        options are handled. """
        return data, labels

    def train_test_split(self, x, y, random_state=42):
        """
        Split x and y data into train/test sets

        Warning: train_test_split() is from sklearn but self.train_test_split()
        is this function, which is what you should use.
        """
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=self.test_percent,
            stratify=y, random_state=random_state)
        return x_train, y_train, x_test, y_test

    def get_file_in_archive(self, archive, filename):
        """ Read one file out of the already-open zip/rar file """
        with archive.open(filename) as fp:
            contents = fp.read()
        return contents

    def create_windows_x(self, x, window_size, overlap):
        """
        Concatenate along dim-1 to meet the desired window_size. We'll skip any
        windows that reach beyond the end. Only process x (saves memory).

        Two options (examples for window_size=5):
            Overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 1,2,3,4,5 and the label of
                example 5
            No overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 5,6,7,8,9 and the label of
                example 9
        """
        x = np.expand_dims(x, axis=1)

        # No work required if the window size is 1, only part required is
        # the above expand dims
        if window_size == 1:
            return x

        windows_x = []
        i = 0

        while i < len(x)-window_size:
            window_x = np.expand_dims(np.concatenate(x[i:i+window_size], axis=0), axis=0)
            windows_x.append(window_x)

            # Where to start the next window
            if overlap:
                i += 1
            else:
                i += window_size

        return np.vstack(windows_x)

    def create_windows_y(self, y, window_size, overlap):
        """
        Concatenate along dim-1 to meet the desired window_size. We'll skip any
        windows that reach beyond the end. Only process y (saves memory).

        Two options (examples for window_size=5):
            Overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 1,2,3,4,5 and the label of
                example 5
            No overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 5,6,7,8,9 and the label of
                example 9
        """
        # No work required if the window size is 1
        if window_size == 1:
            return y

        windows_y = []
        i = 0

        while i < len(y)-window_size:
            window_y = y[i+window_size-1]
            windows_y.append(window_y)

            # Where to start the next window
            if overlap:
                i += 1
            else:
                i += window_size

        return np.hstack(windows_y)

    def create_windows(self, x, y, window_size, overlap):
        """ Split time-series data into windows """
        x = self.create_windows_x(x, window_size, overlap)
        y = self.create_windows_y(y, window_size, overlap)
        return x, y

    def pad_to(self, data, desired_length):
        """
        Pad the number of time steps to the desired length

        Accepts data in one of two formats:
            - shape: (time_steps, features) -> (desired_length, features)
            - shape: (batch_size, time_steps, features) ->
                (batch_size, desired_length, features)
        """
        if len(data.shape) == 2:
            current_length = data.shape[0]
            assert current_length <= desired_length, \
                "Cannot shrink size by padding, current length " \
                + str(current_length) + " vs. desired_length " \
                + str(desired_length)
            return np.pad(data, [(0, desired_length - current_length), (0, 0)],
                    mode="constant", constant_values=0)
        elif len(data.shape) == 3:
            current_length = data.shape[1]
            assert current_length <= desired_length, \
                "Cannot shrink size by padding, current length " \
                + str(current_length) + " vs. desired_length " \
                + str(desired_length)
            return np.pad(data, [(0, 0), (0, desired_length - current_length), (0, 0)],
                    mode="constant", constant_values=0)
        else:
            raise NotImplementedError("pad_to requires 2 or 3-dim data")

    def label_to_int(self, label_name):
        """ e.g. Bathe to 0 """
        return self.class_labels.index(label_name)

    def int_to_label(self, label_index):
        """ e.g. Bathe to 0 """
        return self.class_labels[label_index]


# @register_dataset("uwave")
class uWaveBase(Dataset):
    """
    uWave Gesture dataset

    See: https://zhen-wang.appspot.com/rice/projects_uWave.html

    Either split on days or users:
      - If users: pass days=None, users=[1,2,3,4,5,6,7,8]
      - If days: pass days=[1,2,3,4,5,6,7], users=None
      - To get all data: days=None, users=None
    (or specify any subset of those users/days)
    """
    num_classes = 8
    class_labels = list(range(num_classes))
    feature_names = ["accel_x", "accel_y", "accel_z"]
    users = one_to_n(8)  # 8 people

    def __init__(self, users, *args, days=None, **kwargs):
        self.days = days
        self.users = users
        super().__init__(uWaveBase.num_classes, uWaveBase.class_labels,
            None, None, uWaveBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["uWaveGestureLibrary.zip"],
            "https://zhen-wang.appspot.com/rice/files/uwave")
        return dataset_fp

    def parse_example(self, filename, content):
        """ Load file containing a single example """
        # Get data
        lines = content.decode("utf-8").strip().split("\n")
        data = []

        for line in lines:
            x, y, z = line.split(" ")

            x = float(x)
            y = float(y)
            z = float(z)

            data.append([x, y, z])

        data = np.array(data, dtype=np.float32)

        # Get label from filename
        # Note: there's at least one without a repeat_index, so make it optional
        matches = re.findall(r"[0-9]+-[0-9]*", filename)
        assert len(matches) == 1, \
            "Filename should be in format X_Template_Acceleration#-#.txt but is " \
            + filename + " instead"
        parts = matches[0].split("-")
        assert len(parts) == 2, "Match should be tuple of (gesture index, repeat index)"
        gesture_index, repeat_index = parts
        # The label is the gesture index
        label = int(gesture_index)

        return data, label

    def load_rar(self, filename):
        """ Load RAR file containing examples from one user for one day """
        data = []
        labels = []

        with rarfile.RarFile(filename, "r") as archive:
            filelist = archive.namelist()

            for f in filelist:
                if ".txt" in f and "Acceleration" in f:
                    contents = self.get_file_in_archive(archive, f)
                    new_data, new_label = self.parse_example(f, contents)
                    data.append(new_data)
                    labels.append(new_label)

        return data, labels

    def load_zip(self, filename):
        """ Load ZIP file containing all the RAR files """
        data = []
        labels = []

        with zipfile.ZipFile(filename, "r") as archive:
            filelist = archive.namelist()

            for f in filelist:
                if ".rar" in f:
                    matches = re.findall(r"[0-9]+", f)
                    assert len(matches) == 2, "should be 2 numbers in .rar filename"
                    user, day = matches
                    user = int(user)
                    day = int(day)

                    # Skip data we don't want
                    if self.users is not None:
                        if user not in self.users:
                            #print("Skipping user", user)
                            continue

                    if self.days is not None:
                        if day not in self.days:
                            #print("Skipping day", day)
                            continue

                    #print("Processing user", user, "day", day)
                    contents = self.get_file_in_archive(archive, f)
                    new_data, new_labels = self.load_rar(io.BytesIO(contents))
                    data += new_data
                    labels += new_labels

        # Zero pad (appending zeros) to make all the same shape
        # for uwave_all, we know the max max([x.shape[0] for x in data]) = 315
        # and expand the dimensions to [1, time_steps, num_features] so we can
        # vstack them properly
        #data = [np.expand_dims(self.pad_to(d, 315), axis=0) for d in data]

        #x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        return data, y

    def load(self):
        # Load data
        dataset_fp = self.download()
        x, y = self.load_zip(dataset_fp)
        # Split into train/test sets
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        # Normalize here since we know which data is train vs. test and we have
        # to normalize before we zero pad or the zero padding messes up the
        # mean calculation a lot
        if FLAGS.normalize != "none":
            normalization = calc_normalization_jagged(train_data, FLAGS.normalize)
            train_data = apply_normalization_jagged(train_data, normalization)
            test_data = apply_normalization_jagged(test_data, normalization)

        # Then zero-pad to be the right length
        train_data = np.vstack([np.expand_dims(self.pad_to(d, 315), axis=0)
            for d in train_data]).astype(np.float32)
        test_data = np.vstack([np.expand_dims(self.pad_to(d, 315), axis=0)
            for d in test_data]).astype(np.float32)

        return train_data, train_labels, test_data, test_labels

    def process(self, data, labels):
        """ uWave classes are index-one """
        # Check we have data in [examples, time_steps, 3]
        assert len(data.shape) == 3, "should shape [examples, time_steps, 3]"
        assert data.shape[2] == 3, "should have 3 features"

        # Index one
        labels = labels - 1
        return super().process(data, labels)


#@register_dataset("sleep")
class SleepBase(Dataset):
    """
    Loads sleep RF data files in datasets/RFSleep.zip/*.npy

    Notes:
      - RF data is 30 seconds of data sampled at 25 samples per second, thus
        750 samples. For each of these sets of 750 samples there is a stage
        label.
      - The RF data is complex, so we'll split the complex 5 features into
        the 5 real and then 5 imaginary components to end up with 10 features.
    """
    num_classes = 6
    class_labels = ["Awake", "N1", "N2", "N3", "Light N2", "REM"]
    feature_names = ["real1", "real2", "real3", "real4", "real5",
        "imag1", "imag2", "imag3", "imag4", "imag5"]
    users = zero_to_n(25)  # 26 people

    def __init__(self, users, *args, days=None, **kwargs):
        self.days = days
        self.users = users
        super().__init__(SleepBase.num_classes, SleepBase.class_labels,
            None, None, SleepBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["uWaveGestureLibrary.zip"],
            "https://zhen-wang.appspot.com/rice/files/uwave")
        return dataset_fp

    def process_examples(self, filename, fp):
        d = np.load(fp, allow_pickle=True).item()
        day = int(filename.replace(".npy", ""))
        user = d["subject"]

        # Skip data we don't want
        if self.days is not None:
            if day not in self.days:
                #print("Skipping day", day)
                return None, None, None

        if self.users is not None:
            if user not in self.users:
                #print("Skipping user", user)
                return None, None, None

        #print("Processing user", user, "day", day)

        stage_labels = d["stage"]
        rf = d["rf"]

        # Split 5 complex features into 5 real and 5 imaginary, i.e.
        # now we have 10 features
        rf = np.vstack([np.real(rf), np.imag(rf)])

        assert stage_labels.shape[0]*750 == rf.shape[-1], \
            "If stage labels is of shape (n) then rf should be of shape (5, 750n)"

        # Reshape and transpose into desired format
        x = np.transpose(np.reshape(rf, (rf.shape[0], -1, stage_labels.shape[0])))

        # Drop those that have a label other than 0-5 (sleep stages) since
        # label 6 means "no signal" and 9 means "error"
        no_error = stage_labels < 6
        x = x[no_error]
        stage_labels = stage_labels[no_error]

        assert x.shape[0] == stage_labels.shape[0], \
            "Incorrect first dimension of x (not length of stage labels)"
        assert x.shape[1] == 750, \
            "Incorrect second dimension of x (not 750)"
        assert x.shape[2] == 10, \
            "Incorrect third dimension of x (not 10)"

        return x, stage_labels

    def load_file(self, filename):
        """ Load ZIP file containing all the .npy files """
        if not os.path.exists(filename):
            print("Download unencrypted "+filename+" into the current directory")

        data = []
        labels = []

        with zipfile.ZipFile(filename, "r") as archive:
            filelist = archive.namelist()

            for f in filelist:
                if ".npy" in f:
                    contents = self.get_file_in_archive(archive, f)
                    x, label = self.process_examples(f, io.BytesIO(contents))

                    if x is not None and label is not None:
                        data.append(x)
                        labels.append(label)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        return x, y

    def load(self):
        x, y = self.load_file("RFSleep_unencrypted.zip")
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        return train_data, train_labels, test_data, test_labels


@register_dataset("ucihar")
class UciHarBase(Dataset):
    """
    Loads human activity recognition data files in datasets/UCI HAR Dataset.zip

    Download from:
    https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
    """
    feature_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]
    num_classes = 6
    class_labels = [
        "walking", "walking_upstairs", "walking_downstairs",
        "sitting", "standing", "laying",
    ]
    users = one_to_n(30)  # 30 people
    already_normalized = True

    def __init__(self, users, *args, **kwargs):
        self.users = users
        super().__init__(UciHarBase.num_classes, UciHarBase.class_labels,
            None, None, UciHarBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["UCI%20HAR%20Dataset.zip"],
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00240")
        return dataset_fp

    def get_feature(self, content):
        """
        Read the space-separated, example on each line file

        Returns 2D array with dimensions: [num_examples, num_time_steps]
        """
        lines = content.decode("utf-8").strip().split("\n")
        features = []

        for line in lines:
            features.append([float(v) for v in line.strip().split()])

        return features

    def get_data(self, archive, name):
        """ To shorten duplicate code for name=train or name=test cases """
        def get_data_single(f):
            return self.get_feature(self.get_file_in_archive(archive,
                "UCI HAR Dataset/"+f))

        data = [
            get_data_single(name+"/Inertial Signals/body_acc_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_acc_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_acc_z_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_z_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_z_"+name+".txt"),
        ]

        labels = get_data_single(name+"/y_"+name+".txt")

        subjects = get_data_single(name+"/subject_"+name+".txt")

        data = np.array(data, dtype=np.float32)
        labels = np.squeeze(np.array(labels, dtype=np.float32))
        # Squeeze so we can easily do selection on this later on
        subjects = np.squeeze(np.array(subjects, dtype=np.float32))

        # Transpose from [features, examples, time_steps] to
        # [examples, time_steps (128), features (9)]
        data = np.transpose(data, axes=[1, 2, 0])

        return data, labels, subjects

    def load_file(self, filename):
        """ Load ZIP file containing all the .txt files """
        with zipfile.ZipFile(filename, "r") as archive:
            train_data, train_labels, train_subjects = self.get_data(archive, "train")
            test_data, test_labels, test_subjects = self.get_data(archive, "test")

        all_data = np.vstack([train_data, test_data]).astype(np.float32)
        all_labels = np.hstack([train_labels, test_labels]).astype(np.float32)
        all_subjects = np.hstack([train_subjects, test_subjects]).astype(np.float32)

        # All data if no selection
        if self.users is None:
            return all_data, all_labels

        # Otherwise, select based on the desired users
        data = []
        labels = []

        for user in self.users:
            selection = all_subjects == user
            data.append(all_data[selection])
            current_labels = all_labels[selection]
            labels.append(current_labels)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape)

        return x, y

    def load(self):
        dataset_fp = self.download()
        x, y = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        return train_data, train_labels, test_data, test_labels

    def process(self, data, labels):
        # Index one
        labels = labels - 1
        return super().process(data, labels)


@register_dataset("ucihhar")
class UciHHarBase(Dataset):
    """
    Loads Heterogeneity Human Activity Recognition (HHAR) dataset
    http://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition
    """
    feature_names = [
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
    ]
    num_classes = 6
    class_labels = [
        "bike", "sit", "stand", "walk", "stairsup", "stairsdown",
    ]  # we throw out "null"
    window_size = 128  # to be relatively similar to HAR
    window_overlap = False
    users = zero_to_n(8)  # 9 people

    def __init__(self, users, *args, **kwargs):
        self.users = users
        super().__init__(UciHHarBase.num_classes, UciHHarBase.class_labels,
            UciHHarBase.window_size, UciHHarBase.window_overlap,
            UciHHarBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["Activity%20recognition%20exp.zip"],
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00344/")
        return dataset_fp

    def read_file(self, content):
        """ Read the CSV file """
        lines = content.decode("utf-8").strip().split("\n")
        data_x = []
        data_label = []
        data_subject = []
        users = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

        for line in lines:
            index, arrival, creation, x, y, z, user, \
                model, device, label = line.strip().split(",")

            # Skip the header (can't determine user if invalid)
            if index == "Index":
                continue

            user = users.index(user)  # letter --> number

            # Skip users we don't care about and data without a label
            if user in self.users and label != "null":
                #index = int(index)
                #arrival = float(arrival)
                #creation = float(creation)
                x = float(x)
                y = float(y)
                z = float(z)
                label = self.class_labels.index(label)  # name --> number

                data_x.append((x, y, z))
                data_label.append(label)
                data_subject.append(user)

        data_x = np.array(data_x, dtype=np.float32)
        data_label = np.array(data_label, dtype=np.float32)
        data_subject = np.array(data_subject, dtype=np.float32)

        return data_x, data_label, data_subject

    def get_data(self, archive, name):
        # In their paper, looks like they only did either accelerometer or
        # gyroscope, not aligning them by the creation timestamp. For them the
        # accelerometer data worked better, so we'll just use that for now.
        return self.read_file(self.get_file_in_archive(archive,
                "Activity recognition exp/"+name+"_accelerometer.csv"))

    def load_file(self, filename):
        """ Load ZIP file containing all the .txt files """
        with zipfile.ZipFile(filename, "r") as archive:
            # For now just use phone data since the positions may differ too much
            all_data, all_labels, all_subjects = self.get_data(archive, "Phones")

            # phone_data, phone_labels, phone_subjects = self.get_data(archive, "Phone")
            # watch_data, watch_labels, watch_subjects = self.get_data(archive, "Watch")

        # all_data = np.vstack([phone_data, watch_data]).astype(np.float32)
        # all_labels = np.hstack([phone_labels, watch_labels]).astype(np.float32)
        # all_subjects = np.hstack([phone_subjects, watch_subjects]).astype(np.float32)

        # Otherwise, select based on the desired users
        data = []
        labels = []

        for user in self.users:
            # Load this user's data
            selection = all_subjects == user
            current_data = all_data[selection]
            current_labels = all_labels[selection]
            assert len(current_labels) > 0, "Error: no data for user "+str(user)

            # Split into windows
            current_data, current_labels = self.create_windows(current_data,
                current_labels, self.window_size, self.window_overlap)

            # Save
            data.append(current_data)
            labels.append(current_labels)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape)

        return x, y

    def load(self):
        dataset_fp = self.download()
        x, y = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        return train_data, train_labels, test_data, test_labels


#@register_dataset("ucihm")
class UciHmBase(Dataset):
    """
    Loads sEMG for Basic Hand movements dataset
    http://archive.ics.uci.edu/ml/datasets/sEMG+for+Basic+Hand+movements
    """
    feature_names = [
        "ch1", "ch2",
    ]
    num_classes = 6
    class_labels = [
        "spher", "tip", "palm", "lat", "cyl", "hook",
    ]
    window_size = 500  # 500 Hz, so 1 second
    window_overlap = False  # Note: using np.hsplit, so this has no effect
    users = zero_to_n(5)  # 6 people

    def __init__(self, users, split=True, pad=True, subsample=True,
            *args, **kwargs):
        self.split = split
        # Only apply if split=False
        self.pad = pad
        self.subsample = subsample

        self.users = users
        super().__init__(UciHmBase.num_classes, UciHmBase.class_labels,
            UciHmBase.window_size, UciHmBase.window_overlap,
            UciHmBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["sEMG_Basic_Hand_movements_upatras.zip"],
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00313/")
        return dataset_fp

    def get_data(self, archive, filename):
        """ Open .mat file in zip file, then load contents"""
        with archive.open(filename) as fp:
            mat = scipy.io.loadmat(fp)

            data_x = []
            data_label = []

            for label_index, label in enumerate(self.class_labels):
                # Concatenate the channels
                data = []

                for channel in self.feature_names:
                    data.append(mat[label+"_"+channel])

                # Reshape from (2, 30, 3000) to (30, 3000, 2) for 30 examples,
                # 3000 time steps, and 2 features
                data = np.array(data, dtype=np.float32).transpose([1, 2, 0])

                # Split from 3000 or 2500 time steps into 6 or 5 non-overlapping
                # windows of 500 samples. Duplicate the label 6 or 5 times,
                # respectively.
                if self.split:
                    assert data.shape[1] in [2500, 3000], "data.shape[1] should " \
                        "be 2500 or 3000, but is "+str(data.shape[1])
                    num_windows = data.shape[1]//self.window_size
                    data_x += np.hsplit(data, num_windows)
                else:
                    # Since we have to concatenate multiple domains, pad with
                    # zeros to get them to all be the same number of time steps
                    if self.pad:
                        data = self.pad_to(data, 3000)

                    # It's really slow with 3000 samples, and 500 Hz is probably
                    # overkill, so subsample
                    if self.subsample:
                        data = data[:, ::6, :]  # 3000 -> 500, fewer samples

                    data_x.append(data)
                    num_windows = 1

                data_label += [label_index]*len(data)*num_windows

            data_x = np.vstack(data_x).astype(np.float32)
            data_label = np.array(data_label, dtype=np.float32)

            return data_x, data_label

    def load_file(self, filename):
        """
        Load desired participants' data

        Numbering:
            0 - Database 1/female_1.mat
            1 - Database 1/female_2.mat
            2 - Database 1/female_3.mat
            3 - Database 1/male_1.mat
            4 - Database 1/male_2.mat
            5 - Database 2/male_day_{1,2,3}.mat
        """
        data = []
        labels = []

        with zipfile.ZipFile(filename, "r") as archive:
            for user in self.users:
                # Load this user's data
                if user != 5:
                    gender = "female" if user < 3 else "male"
                    index = user+1 if user < 3 else user-2
                    current_data, current_labels = self.get_data(archive,
                        "Database 1/"+gender+"_"+str(index)+".mat")
                else:
                    data1, labels1 = self.get_data(archive, "Database 2/male_day_1.mat")
                    data2, labels2 = self.get_data(archive, "Database 2/male_day_2.mat")
                    data3, labels3 = self.get_data(archive, "Database 2/male_day_3.mat")
                    current_data = np.vstack([data1, data2, data3]).astype(np.float32)
                    current_labels = np.hstack([labels1, labels2, labels3]).astype(np.float32)

                # Save
                data.append(current_data)
                labels.append(current_labels)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape)

        return x, y

    def load(self):
        dataset_fp = self.download()
        x, y = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        return train_data, train_labels, test_data, test_labels


class WisdmBase(Dataset):
    """
    Base class for the WISDM datasets
    http://www.cis.fordham.edu/wisdm/dataset.php
    """
    feature_names = [
        "acc_x", "acc_y", "acc_z",
    ]
    window_size = 128  # similar to HAR
    window_overlap = False

    def __init__(self, users, num_classes, class_labels, *args, **kwargs):
        self.users = users
        super().__init__(num_classes, class_labels,
            WisdmBase.window_size, WisdmBase.window_overlap,
            WisdmBase.feature_names, *args, **kwargs)
        # Override and set these
        #self.filename_prefix = ""
        #self.download_filename = ""

    def download(self):
        (dataset_fp,) = self.download_dataset([self.download_filename],
            "http://www.cis.fordham.edu/wisdm/includes/datasets/latest/")
        return dataset_fp

    def read_data(self, lines, user_list):
        """ Read the raw data CSV file """
        data_x = []
        data_label = []
        data_subject = []

        for line in lines:
            parts = line.strip().replace(";", "").split(",")

            # For some reason there's blank rows in the data, e.g.
            # a bunch of lines like "577,,;"
            # Though, allow 7 since sometimes there's an extra comma at the end:
            # "21,Jogging,117687701514000,3.17,9,1.23,;"
            if len(parts) != 6 and len(parts) != 7:
                continue

            # Skip if x, y, or z is blank
            if parts[3] == "" or parts[4] == "" or parts[5] == "":
                continue

            user = int(parts[0])

            # Skip users that may not have enough data
            if user in user_list:
                user = user_list.index(user)  # non-consecutive to consecutive

                # Skip users we don't care about
                if user in self.users:
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    label = self.class_labels.index(parts[1])  # name --> number

                    data_x.append((x, y, z))
                    data_label.append(label)
                    data_subject.append(user)

        data_x = np.array(data_x, dtype=np.float32)
        data_label = np.array(data_label, dtype=np.float32)
        data_subject = np.array(data_subject, dtype=np.float32)

        return data_x, data_label, data_subject

    def read_user_list(self, lines, min_test_samples=30):
        """ Read first column of the CSV file to get a unique list of uid's
        Also, skip users with too few samples """
        user_sample_count = {}

        for line in lines:
            parts = line.strip().split(",")

            # There's some lines without the right number of parts, e.g. blank
            if len(parts) != 6 and len(parts) != 7:
                continue

            # Skip if x, y, or z is blank
            if parts[3] == "" or parts[4] == "" or parts[5] == "":
                continue

            uid = int(parts[0])

            # There are duplicates in the file for some reason (so, either the
            # same person or it's not truly unique)
            if uid not in user_sample_count:
                user_sample_count[uid] = 0
            else:
                user_sample_count[uid] += 1

        # Remove users with too few samples
        user_list = []

        # How many samples we need: to stratify the sklearn function says
        # The test_size = A should be greater or equal to the number of classes = B
        # x/128*.2 > 6 classes
        # x > 6*128/.2
        # Though, actually, just set the minimum test samples. It's probably not
        # enough to have only 7...
        test_percentage = 0.20  # default
        #min_samples = int(len(self.class_labels)*self.window_size/test_percentage)
        min_samples = int(min_test_samples*self.window_size/test_percentage)

        for user, count in user_sample_count.items():
            if count > min_samples:
                user_list.append(user)

        # Data isn't sorted by user in the file
        user_list.sort()

        return user_list

    def get_lines(self, archive, name):
        """ Open and load file in tar file, get lines from file """
        f = archive.extractfile(self.filename_prefix+name)

        if f is None:
            return None

        return f.read().decode("utf-8").strip().split("\n")

    def load_file(self, filename):
        """ Load desired participants' data """
        # Get data
        with tarfile.open(filename, "r") as archive:
            raw_data = self.get_lines(archive, "raw.txt")

        # Some of the data doesn't have a uid in the demographics file? So,
        # instead just get the user list from the raw data. Also, one person
        # have very little data, so skip them (e.g. one person only has 25
        # samples, which is only 0.5 seconds of data -- not useful).
        user_list = self.read_user_list(raw_data)

        #print("Number of users:", len(user_list))

        # For now just use phone data since the positions may differ too much
        all_data, all_labels, all_subjects = self.read_data(raw_data, user_list)

        # Otherwise, select based on the desired users
        data = []
        labels = []

        for user in self.users:
            # Load this user's data
            selection = all_subjects == user
            current_data = all_data[selection]
            current_labels = all_labels[selection]
            assert len(current_labels) > 0, "Error: no data for user "+str(user)

            # Split into windows
            current_data, current_labels = self.create_windows(current_data,
                current_labels, self.window_size, self.window_overlap)

            # Save
            data.append(current_data)
            labels.append(current_labels)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape)

        return x, y

    def load(self):
        dataset_fp = self.download()
        x, y = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        return train_data, train_labels, test_data, test_labels


@register_dataset("wisdm_at")
class WisdmAtBase(WisdmBase):
    """
    Loads Actitracker dataset
    http://www.cis.fordham.edu/wisdm/dataset.php#actitracker (note: click
    on Actitracker link on left menu)
    """
    num_classes = 6
    class_labels = [
        "Walking", "Jogging", "Stairs", "Sitting", "Standing", "LyingDown",
    ]
    users = zero_to_n(50)  # 51 people

    def __init__(self, users, *args, **kwargs):
        self.filename_prefix = "home/share/data/public_sets/WISDM_at_v2.0/WISDM_at_v2.0_"
        self.download_filename = "WISDM_at_latest.tar.gz"
        super().__init__(users,
            WisdmAtBase.num_classes, WisdmAtBase.class_labels, *args, **kwargs)


@register_dataset("wisdm_ar")
class WisdmArBase(WisdmBase):
    """
    Loads WISDM Activity prediction/recognition dataset
    http://www.cis.fordham.edu/wisdm/dataset.php
    """
    num_classes = 6
    class_labels = [
        "Walking", "Jogging", "Sitting", "Standing", "Upstairs", "Downstairs",
    ]
    users = zero_to_n(32)  # 33 people

    def __init__(self, users, *args, **kwargs):
        self.filename_prefix = "WISDM_ar_v1.1/WISDM_ar_v1.1_"
        self.download_filename = "WISDM_ar_latest.tar.gz"
        super().__init__(users,
            WisdmArBase.num_classes, WisdmArBase.class_labels, *args, **kwargs)


class WatchBase(Dataset):
    """
    Our watch dataset

    Doesn't actually load the dataset, since we create the tfrecord files in
    watch-protobuf/process/*.py (see separate git repo). This just provides
    the information about it.
    """
    feature_names = [
        # Time features
        "second", "minute", "hour12", "hour24", "since_midnight", "weekday",
        "day", "day_of_year", "month", "year",

        # One-hot-encoded location category (and "other")
        "cat_amenity", "cat_building", "cat_footway", "cat_highway",
        "cat_leisure", "cat_service", "cat_shop", "cat_tourism",
        "cat_water", "cat_other",

        # One-hot-encoded location sub-category (and "other")
        "subcat_convenience", "subcat_crossing", "subcat_driveway",
        "subcat_hotel", "subcat_information", "subcat_lake", "subcat_park",
        "subcat_parking", "subcat_parking_aisle", "subcat_pitch", "subcat_pond",
        "subcat_reservoir", "subcat_residential", "subcat_service",
        "subcat_sidewalk", "subcat_swimming_pool", "subcat_track", "subcat_yes",
        "subcat_other",

        # Raw features
        "roll", "pitch", "yaw",
        "rot_rate_x", "rot_rate_y", "rot_rate_z",
        "user_acc_x", "user_acc_y", "user_acc_z",
        "grav_x", "grav_y", "grav_z",
        #"heading",
        "acc_x", "acc_y", "acc_z",
        #"long", "lat", "horiz_acc",
        "alt",
        #"vert_acc",
        "course", "speed",
        #"floor",
    ]
    window_size = 128  # similar to HAR
    window_overlap = False

    def __init__(self, users, num_classes, class_labels, *args, **kwargs):
        self.users = users
        super().__init__(num_classes, class_labels,
            WatchBase.window_size, WatchBase.window_overlap,
            WatchBase.feature_names, *args, **kwargs)

    def load(self):
        return None, None, None, None


#@register_dataset("watch_withother")
class WatchWithOther(WatchBase):
    num_classes = 7
    class_labels = [
        "Cook", "Eat", "Hygiene", "Work", "Exercise", "Travel", "Other",
    ]
    users = one_to_n(15)  # 15 people

    def __init__(self, users, *args, **kwargs):
        super().__init__(users, WatchWithOther.num_classes,
            WatchWithOther.class_labels, *args, **kwargs)


# @register_dataset("watch_noother")
class WatchWithoutOther(WatchBase):
    num_classes = 6
    class_labels = [
        "Cook", "Eat", "Hygiene", "Work", "Exercise", "Travel",
    ]
    users = one_to_n(15)  # 15 people

    def __init__(self, users, *args, **kwargs):
        super().__init__(users, WatchWithoutOther.num_classes,
            WatchWithoutOther.class_labels, *args, **kwargs)


# Get datasets
def load(dataset_name_to_load, *args, **kwargs):
    """ Load a dataset based on the name (must be one of datasets.names()) """
    dataset_class = None
    dataset_object = None

    # Go through list of valid datasets, create the one this matches
    for name in list_datasets():
        for user in get_dataset_users(name):
            dataset_name = name+"_"+str(user)

            if dataset_name_to_load == dataset_name:
                dataset_class = get_dataset(name)
                dataset_object = call_dataset(name, users=[user],
                    *args, **kwargs)
                break

    if dataset_object is None:
        raise NotImplementedError("unknown dataset "+dataset_name_to_load)

    return dataset_object, dataset_class


# Get attributes: num_classes, class_labels (required in load_datasets.py)
def attributes(dataset_name_to_load):
    """ Get num_classes, class_labels for dataset (must be one of datasets.names()) """
    num_classes = None
    class_labels = None

    # Go through list of valid datasets, load attributes of the one this matches
    for name in list_datasets():
        for user in get_dataset_users(name):
            dataset_name = name+"_"+str(user)

            if dataset_name_to_load == dataset_name:
                d = get_dataset(name)
                num_classes = d.num_classes
                class_labels = d.class_labels
                break

    return num_classes, class_labels


# List of all valid dataset names
def names():
    """ Returns list of all the available datasets to load with
    datasets.load(name) """
    datasets = []

    for name in list_datasets():
        for user in get_dataset_users(name):
            datasets.append(name+"_"+str(user))

    return datasets


def main(argv):
    sd = load("ucihar_1")

    print("Source dataset")
    print(sd.train_data, sd.train_labels)
    print(sd.train_data.shape, sd.train_labels.shape)
    print(sd.test_data, sd.test_labels)
    print(sd.test_data.shape, sd.test_labels.shape)