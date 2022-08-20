MIMIC-III Data Preprocessing
=========================




## Requirements

We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. Otherwise, generally we make liberal use of the following packages:

- numpy
- pandas


## Building datasets

Here are the required steps to build the benchmark. It assumes that you already have MIMIC-III dataset (lots of CSV files) on the disk.
1. Clone the repo.

       git clone https://github.com/YerevaNN/mimic3-benchmarks/
    
2. The following command takes MIMIC-III CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`. This step might take around an hour.

       python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/

3. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows (more information can be found in [`mimic3benchmark/scripts/more_on_validating_events.md`](mimic3benchmark/scripts/more_on_validating_events.md)).

       python -m mimic3benchmark.scripts.validate_events data/root/

4. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```{SUBJECT_ID}/episode{#}.csv```. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). **Outlier detection is disabled in the current version**.

       python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/

5. The next command splits the whole dataset into training and testing sets. Note that the train/test split is the same of all tasks.

       python -m mimic3benchmark.scripts.split_train_and_test data/root/
	
6. The following commands will generate modality-specific datasets, which can later be used to train models. These commands are independent, if you are going to work only on one benchmark modality, you can run only the corresponding command.

       python -m mimic3benchmark.scripts.create_timeseries data/root/ data/timeseries/
       python -m mimic3benchmark.scripts.create_demography_diagnosis data/root/ data/demography_diagnosis/
       python -m mimic3benchmark.scripts.create_clinical_notes --mimic_dir {mimiciii directory} --save_dir data/clinical_notes/ --admission_only True


After the above commands are done, there will be a directory `data/{modality}` for each created benchmark task.
These directories have two sub-directories: `train` and `test`.
Each of them contains bunch of ICU stays and one file with name `listfile.csv`, which lists all samples in that particular set.
Each row of `listfile.csv` has the following form: `stay,HADM_ID,label(s)`.
In in-hospital mortality prediction task period_length is always 48 hours, so it is not listed in corresponding listfiles.


### Train / validation split

Use the following command to extract validation set from the training set. This step is required for running the baseline models. Likewise the train/test split, the train/validation split is the same for all tasks.

       python -m mimic3benchmark.scripts.split_train_val {dataset-directory}
       
`{dataset-directory}` can be either `data/timeseries`, `data/demography&diagnosis`, or `data/clinical_notes`.

## Citation

This repository is based on https://github.com/YerevaNN/mimic3-benchmarks/blob/master/README.md and https://github.com/bvanaken/clinical-outcome-prediction. Really appreciate the work from both repos!