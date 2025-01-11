# ECG Dataset Preparation

Notes on .ipymn files (code is not cleaned)

ECG file (dataset) is required (It can be obtained from physionet)

## Done
If it has been preprocessed, an additional folder will be created in this location and the data will be stored there.

for example, with ptbxl data (downstream dataset):
- ./ptbxl/ecgs500: the preprocessed ECG signals
- ./ptbxl/form: csv file containing various information (classes, storage location, etc.) of the train, valid, and test datasets for a super task.
- ./ptbxl/form: csv file containing various information (classes, storage location, etc.) of the train, valid, and test datasets for a sub task.
- ./ptbxl/form: csv file containing various information (classes, storage location, etc.) of the train, valid, and test datasets for a form task.
- ./ptbxl/form: csv file containing various information (classes, storage location, etc.) of the train, valid, and test datasets for a rhythm task.
- The csv files related index for each task is referenced [here](https://github.com/cheliu-computation/MERL-ICML2024/tree/main/finetune/data_split/ptbxl)

for example, with chapman data (pre-trained dataset):
- ./chapman/ecgs: the preprocessed ECG signals
- ./chapman/index.csv: csv file to check where ecg signal files are stored
