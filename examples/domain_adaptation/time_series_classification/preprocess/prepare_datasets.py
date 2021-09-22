#!/usr/bin/env python3
"""
Process each dataset into .tfrecord files

Run (or see ../generate_tfrecords.sh script):

    python -m datasets.main <args>

Note: probably want to run this prefixed with CUDA_VISIBLE_DEVICES= so that it
doesn't use the GPU (if you're running other jobs). Does this by default if
parallel=True since otherwise it'll error.
"""
import os
import numpy as np
import tensorflow as tf
import torch
import tqdm
import multiprocessing

from absl import app
from absl import flags
from sklearn.model_selection import train_test_split

import sys
sys.path.append('.')

import datasets
from normalization import apply_normalization, calc_normalization

FLAGS = flags.FLAGS

flags.DEFINE_boolean("parallel", True, "Run multiple in parallel")
flags.DEFINE_integer("jobs", 0, "Parallel jobs (if parallel=True), 0 = # of CPU cores")
flags.DEFINE_boolean("debug", False, "Whether to print debug information")


def run_job_pool(func, argsList, desc=None, cores=None):
    """
    Processor pool to use multiple cores, with a progress bar

    func = function to execute
    argsList = array of tuples, each tuple is the arguments to pass to the function

    Combination of:
    https://stackoverflow.com/a/43714721/2698494
    https://stackoverflow.com/a/45652769/2698494

    Returns:
    an array of the outputs from the function

    Example:
    # Define a function that'll be run a bunch of times
    def f(a,b):
        return a+b

    # Array of arrays (or tuples) of the arguments for the function
    commands = [[1,2],[3,4],[5,6],[7,8]]
    results = run_job_pool(f, commands, desc="Addition")
    """
    if cores is None:
        p = multiprocessing.Pool(multiprocessing.cpu_count())
    else:
        p = multiprocessing.Pool(cores)
    processes = []
    results = []

    for args in argsList:
        processes.append(p.apply_async(func, args))

    with tqdm.tqdm(total=len(processes), desc=desc) as pbar:
        for process in processes:
            results.append(process.get())
            pbar.update()
    pbar.close()
    p.close()
    p.join()

    return results


def write(filename, x, y):
    if x is not None and y is not None:
        if not os.path.exists(filename):
            print("Writing:", filename)
            data = {
                'x': torch.tensor(x),
                'y': torch.tensor(y)
            }
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(data, filename)
            # write_tfrecord(filename, x, y)
        elif FLAGS.debug:
            print("Skipping:", filename, "(already exists)")
    elif FLAGS.debug:
        print("Skipping:", filename, "(no data)")


def shuffle_together_calc(length, seed=None):
    """ Generate indices of numpy array shuffling, then do x[p] """
    rand = np.random.RandomState(seed)
    p = rand.permutation(length)
    return p


def to_numpy(value):
    """ Make sure value is numpy array """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return value


def valid_split(data, labels, seed=None, validation_size=1000):
    """ (Stratified) split training data into train/valid as is commonly done,
    taking 1000 random (stratified) (labeled, even if target domain) samples for
    a validation set """
    percentage_size = int(0.2*len(data))
    if percentage_size > validation_size:
        test_size = validation_size
    else:
        if FLAGS.debug:
            print("Warning: using smaller validation set size", percentage_size)
        test_size = 0.2  # 20% maximum

    x_train, x_valid, y_train, y_valid = \
        train_test_split(data, labels, test_size=test_size,
            stratify=labels, random_state=seed)

    return x_valid, y_valid, x_train, y_train


def save_dataset(dataset_name, output_dir, seed=0):
    """ Save single dataset """
    train_filename = os.path.join(output_dir,
        dataset_name, "train.pth")
    valid_filename = os.path.join(output_dir, dataset_name, "valid.pth")
    test_filename = os.path.join(output_dir, dataset_name, "test.pth")

    # Skip if they already exist
    if os.path.exists(train_filename) \
            and os.path.exists(valid_filename) \
            and os.path.exists(test_filename):
        if FLAGS.debug:
            print("Skipping:", train_filename, valid_filename, test_filename,
               "already exist")
        return

    if FLAGS.debug:
        print("Saving dataset", dataset_name)
    dataset, dataset_class = datasets.load(dataset_name)

    # Skip if already normalized/bounded, e.g. UCI HAR datasets
    already_normalized = dataset_class.already_normalized

    # Split into training/valid datasets
    valid_data, valid_labels, train_data, train_labels = \
        valid_split(dataset.train_data, dataset.train_labels, seed=seed)

    # Calculate normalization only on the training data
    if FLAGS.normalize != "none" and not already_normalized:
        normalization = calc_normalization(train_data, FLAGS.normalize)

        # Apply the normalization to the training, validation, and testing data
        train_data = apply_normalization(train_data, normalization)
        valid_data = apply_normalization(valid_data, normalization)
        test_data = apply_normalization(dataset.test_data, normalization)
    else:
        test_data = dataset.test_data

    # Saving
    write(train_filename, train_data, train_labels)
    write(valid_filename, valid_data, valid_labels)
    write(test_filename, test_data, dataset.test_labels)


def main(argv):
    output_dir = os.path.join("../data")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all possible datasets we can generate
    adaptation_problems = datasets.names()

    # Save tfrecord files for each of the adaptation problems
    if FLAGS.parallel:
        # TensorFlow will error from all processes trying to use ~90% of the
        # GPU memory on all parallel jobs, which will fail, so do this on the
        # CPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if FLAGS.jobs == 0:
            cores = None
        else:
            cores = FLAGS.jobs

        run_job_pool(save_dataset,
            [(d, output_dir) for d in adaptation_problems], cores=cores)
    else:
        for dataset_name in adaptation_problems:
            save_dataset(dataset_name, output_dir)


if __name__ == "__main__":
    app.run(main)
