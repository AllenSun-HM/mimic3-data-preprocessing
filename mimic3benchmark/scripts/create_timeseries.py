from __future__ import absolute_import
from __future__ import print_function

import json
import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm


def process_partition(args, partition, eps=1e-6, n_hours=48):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    mp_listfile = pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID"])

    with open("mimic3benchmark/resources/channel_info.json") as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    with open("mimic3benchmark/resources/discretizer_config.json") as discretizer_config_file:
        discretizer_config = json.loads(discretizer_config_file.read())



    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        patient_stays_df = pd.read_csv(patient_folder+"/stays.csv")

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "") # the name of episode data (for example episode1.csv)
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
                # empty label file
                if label_df.shape[0] == 0:
                    continue

                mortality = int(label_df.iloc[0]["Mortality"])
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                if los < n_hours - eps:
                    continue

                ts_df = pd.read_csv(os.path.join(patient_folder, ts_filename))
                ts_df = ts_df[(ts_df['Hours'] > -eps) & (ts_df['Hours'] < n_hours + eps)]

                event_times = ts_df['Hours'].to_numpy()

                # no measurements in ICU
                if len(event_times) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                for col in ts_df.columns:
                    if col == 'Hours':
                        continue
                    if discretizer_config['is_categorical_channel'][col]:
                        not_na_indice = ts_df[col].notna()
                        ts_df[col][not_na_indice] = ts_df[col][not_na_indice].map(channel_info[col]['values'])



                output_ts_filename = patient + "_" + ts_filename

                subject_id, hadm_id = patient_stays_df['SUBJECT_ID'][0], patient_stays_df[patient_stays_df['ICUSTAY_ID'] == label_df['Icustay'].values[0]]['HADM_ID'].values[0]
                ts_df['HADM_ID'] = hadm_id
                ts_df.to_csv(os.path.join(output_dir, output_ts_filename))
                mp_listfile = mp_listfile.append({'SUBJECT_ID': subject_id, 'HADM_ID': hadm_id}, ignore_index=True)
                xy_pairs.append((output_ts_filename, mortality, hadm_id))

    print("Number of created samples:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,y_true,HADM_ID\n')
        for (x, y, z) in xy_pairs:
            listfile.write('{},{:d},{:d}\n'.format(x, y, z))
    mp_listfile.to_csv('./mp_listfile.csv')

def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
