from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import yaml
import random
random.seed(49297)
from tqdm import tqdm


def process_partition(args, definitions, code_to_group, id_to_group, group_to_id,
                      partition, eps=1e-6):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    rows = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
                patient_stays_df = pd.read_csv(patient_folder + "/stays.csv")

                # empty label file
                if label_df.shape[0] == 0:
                    continue

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                ts_lines = tsfile.readlines()
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                if los < 48 - eps:
                    continue


                cur_labels = [0 for i in range(len(id_to_group))]

                age = label_df['Age'].iloc[0]
                gender = label_df['Gender'].iloc[0]
                male = int(gender == 2)
                female = int(gender == 1)
                mortality = label_df['Mortality'].iloc[0]
                icustay = label_df['Icustay'].iloc[0]
                diagnoses_df = pd.read_csv(os.path.join(patient_folder, "diagnoses.csv"),
                                           dtype={"ICD9_CODE": str})
                diagnoses_df = diagnoses_df[diagnoses_df.ICUSTAY_ID == icustay]
                for index, row in diagnoses_df.iterrows():
                    if row['USE_IN_BENCHMARK']:
                        code = row['ICD9_CODE']
                        group = code_to_group[code]
                        group_id = group_to_id[group]
                        cur_labels[group_id] = 1

                cur_labels = [x for (i, x) in enumerate(cur_labels)
                              if definitions[id_to_group[i]]['use_in_benchmark']]
                subject_id, hadm_id = patient_stays_df['SUBJECT_ID'][0], patient_stays_df[patient_stays_df['ICUSTAY_ID'] == label_df['Icustay'].values[0]]['HADM_ID'].values[0]
                rows.append((hadm_id, mortality, los, age, male, female, cur_labels))

    print("Number of created samples:", len(rows))
    if partition == "train":
        random.shuffle(rows)
    if partition == "train":
        rows = sorted(rows)

    codes_in_benchmark = [x for x in id_to_group
                          if definitions[x]['use_in_benchmark']]

    listfile_header = "hadm_id,mortality,period_length,age,male,female," + ",".join(codes_in_benchmark)
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write(listfile_header + "\n")
        for (hadm_id, mortality, t, age, male, female, y) in rows:
            labels = ','.join(map(str, y))
            listfile.write('{},,{},{:.6f},{}, {}, {}, {}\n'.format(hadm_id, mortality, t, age, male, female, labels))


def main():
    parser = argparse.ArgumentParser(description="Create data for phenotype classification task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    parser.add_argument('--phenotype_definitions', '-p', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccs_2015_definitions.yaml'),
                        help='YAML file with phenotype definitions.')
    args, _ = parser.parse_known_args()

    with open(args.phenotype_definitions) as definitions_file:
        definitions = yaml.safe_load(definitions_file)

    code_to_group = {}
    for group in definitions:
        codes = definitions[group]['codes']
        for code in codes:
            if code not in code_to_group:
                code_to_group[code] = group
            else:
                assert code_to_group[code] == group

    id_to_group = sorted(definitions.keys())
    group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, definitions, code_to_group, id_to_group, group_to_id, "test")
    process_partition(args, definitions, code_to_group, id_to_group, group_to_id, "train")


if __name__ == '__main__':
    main()
