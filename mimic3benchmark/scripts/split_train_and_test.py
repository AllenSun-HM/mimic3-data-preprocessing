from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse

import pandas as pd


def move_to_partition(args, patients, partition):
    if not os.path.exists(os.path.join(args.subjects_root_path, partition)):
        os.mkdir(os.path.join(args.subjects_root_path, partition))
    for patient in patients:
        src = os.path.join(args.subjects_root_path, patient)
        dest = os.path.join(args.subjects_root_path, partition, patient)
        shutil.move(src, dest)


def main():
    parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    args, _ = parser.parse_known_args()
    folders = os.listdir(args.subjects_root_path)
    folders = list((filter(str.isdigit, folders)))

    test_set = set(pd.read_csv('mimic3benchmark/mimic_test.csv')['SUBJECT_ID'].values)
    train_patients = [x for x in folders if int(x) not in test_set]
    test_patients = [x for x in folders if int(x) in test_set]
    assert len(set(train_patients) & set(test_patients)) == 0

    move_to_partition(args, train_patients, "train")
    move_to_partition(args, test_patients, "test")


if __name__ == '__main__':
    main()
