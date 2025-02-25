import pyedflib
import numpy as np 
import pandas as pd
import os
from scipy import signal

def load_edf_file(file_path, patient, xiao_channels):
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    if not xiao_channels:
        if n == 22: 
            return pd.DataFrame()
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    f.close()
    df_signals = pd.DataFrame(sigbufs).transpose()
    df_signals.columns = signal_labels
    if not xiao_channels:
        if patient in n_channels_28 and n!=23: 
            df_signals = df_signals.drop(df_signals.columns[channels_to_drop], axis=1)

    if xiao_channels: 
            xiao_channels_sel = ['FZ-CZ', 'CZ-PZ', 'F8-T8', 'P4-O2', 'FP2-F8', 'F4-C4', 
                                 'C4-P4', 'P3-O1', 'FP2-F4', 'F3-C3', 'C3-P3', 'P7-O1', 
                                 'FP1-F3', 'F7-T7', 'T7-P7','FP1-F7']
            try: 
                df_signals = df_signals[xiao_channels_sel]
            except KeyError: 
                print(f"In recording {file_path} channel names are {signal_labels}, skipping the file completely")
                df_signals = pd.DataFrame()
        
    return df_signals

def load_seizure_records(file_path):
    with open(file_path) as f:
        seizure_records = f.read().splitlines()
    return seizure_records


def create_labels(df_annotations, file_name, total_samples): 
    seizure_info = df_annotations[df_annotations['Sub File'].str.startswith(file_name)]
    labels = np.zeros(total_samples)
    for _, row in seizure_info.iterrows():
        start = int(row['Ictal Start Row'])
        end = int(row['Ictal End Row'])
        labels[start:end] = 1
    return labels

def preprocess_data(df_signals, sampling_rate=256):
    # Apply bandpass filter (0.5 - 25 Hz)
    #In other papers they use different bandpass filters up to 45, 60 etc 
    sos = signal.butter(4, [0.5, 45], btype='bandpass', fs=sampling_rate, output='sos')
    filtered_signals = signal.sosfilt(sos, df_signals.values.T).T
    return filtered_signals


project_path = "./data/raw/CHB-MIT"
annotation_file = os.path.join(project_path, "CHB-MIT DB timestamp.csv")
seizure_records_file = os.path.join(project_path,"RECORDS-WITH-SEIZURES")
output_path_all = "./data/processed/CHB-MIT/"
downsample = True
xiao_channels = True
df_annotations = pd.read_csv(annotation_file)

#all channels are used 
n_channels_23 = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09",
    "chb10", "chb23", "chb24"]
n_channels_28 = ["chb13", "chb15", "chb16", "chb14",  "chb11", "chb12", "chb17", "chb18", "chb19", "chb20", "chb21", "chb22"]

channels_to_drop = [4, 9, 12, 17, 22]


seizure_records = load_seizure_records(seizure_records_file)
selected_patients =  n_channels_28 + n_channels_23

for patient in selected_patients: 
    print(patient)
    patient_data = []
    patient_labels = []
    if xiao_channels: 
        output_path_all = "./data/processed/CHB-MIT/biclass"

    patient_files = [f for f in seizure_records if f.startswith(patient)]
    output_path = os.path.join(output_path_all, patient)
    os.makedirs(output_path, exist_ok=True)
    for file_path in patient_files:
        file_name = file_path.split('/')[-1].split('.')[0]
        print(file_name, file_path)
        edf_file = os.path.join(project_path, file_path)

        df_signals = load_edf_file(edf_file, patient, xiao_channels)
        if df_signals.empty: 
            continue
        filtered_signals = preprocess_data(df_signals)
        labels = create_labels(df_annotations, file_name, len(df_signals))

        # Downsample to 1 second resolution
        #What are other downsampling methods? 
        if downsample: 
            filtered_signals = filtered_signals[::2] #TODO: calculate from filter by Nyquist
            labels = labels[::2]  #TODO: calculate from filter by Nyquist 128

        patient_data.append(filtered_signals)
        patient_labels.append(labels)


    patient_data = np.concatenate(patient_data, axis=0)
    patient_labels = np.concatenate(patient_labels, axis=0)

    np.save(os.path.join(output_path, f"{patient}_data.npy"), patient_data)
    np.save(os.path.join(output_path, f"{patient}_labels.npy"), patient_labels)
