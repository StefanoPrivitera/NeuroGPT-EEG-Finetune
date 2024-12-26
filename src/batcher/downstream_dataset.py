import os
from pathlib import Path
import mne
import numpy as np
from batcher.base import EEGDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from eremus.eremus_utils import getPrunedSessions, sub
from eremus import gew
import torch

import pandas as pd
class MotorImageryDataset(EEGDataset):
    def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.data_all = []
        for fn in self.filenames:
            if fn.endswith('.gdf'):
                print(f"Loading GDF file: {fn}")
                raw = mne.io.read_raw_gdf(fn, preload=True)
                data, times = raw[:]
                data_dict = {
                    's': data,
                    'etyp': raw.annotations.description,
                    'epos': raw.annotations.onset * raw.info['sfreq'],
                    'edur': raw.annotations.duration * raw.info['sfreq'],
                    'artifacts': np.zeros_like(raw.annotations.onset)  # Modify as needed
                }
                self.data_all.append(data_dict)
                print(f"Loaded GDF file: {fn} with data shape {data.shape}")
                print(f"Annotations: {raw.annotations}")
            else:
                print(f"Loading NPY file: {fn}")
                data_dict = np.load(fn, allow_pickle=True).item()
                self.data_all.append(data_dict)
                # print(f"Loaded NPY file: {fn} with data: {data_dict}")
                print(f"Loaded NPY file: {fn}")

        # Debugging: Check structure of data_all
        print(f"data_all structure: {[type(item) for item in self.data_all]}")
        # print(f"Example data structure: {self.data_all[0]}")

        self.mi_types = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 1023: 'rejected'}
        self.labels_string2int = {'left': 0, 'right': 1, 'foot': 2, 'tongue': 3}
        self.Fs = 250  # 250Hz from original paper

        # Load transformation matrix and ensure correct shape
        self.P = np.load("../inputs/tMatrix_value.npy")
        if self.P.shape != (22, 22):
            raise ValueError("Transformation matrix self.P must be 22x22.")

        self.trials, self.labels, self.num_trials_per_sub = self.get_trials_all()

    def __len__(self):
        total_len = sum(self.num_trials_per_sub)
        # print(f"Dataset length: {total_len}")
        return total_len

    def __getitem__(self, idx):
        # print(f"Accessing index: {idx} of {self.__len__()}")  # Debugging statement
        if idx >= len(self.trials) or idx < 0:
            raise IndexError(f"Index {idx} out of bounds for trials of size {len(self.trials)}")
        return self.preprocess_sample(self.trials[idx], self.num_chunks, self.labels[idx])

    def map2pret(self, data):
        # Apply transformation matrix to align channel configuration
        print(f"Shape of self.P: {self.P.shape}")  # Debugging statement
        print(f"Shape of data before transformation: {data.shape}")  # Debugging statement

        if data.shape[1] != 22:
            raise ValueError("Mismatch between data channels and transformation matrix channels.")

        # Perform transformation on (batch, channels, time) format
        transformed_data = np.einsum('ij,bjt->bit', self.P, data)
        print(f"Shape of data after transformation: {transformed_data.shape}")
        
        return transformed_data

    def get_trials_from_single_subj(self, sub_id):
        try:
            raw = self.data_all[sub_id]['s']  # (channels, time)
            events_type = self.data_all[sub_id]['etyp']
            events_position = self.data_all[sub_id]['epos']
            events_duration = self.data_all[sub_id]['edur']
            artifacts = self.data_all[sub_id]['artifacts']

            # print(f"Subject {sub_id} annotations: {events_type}")

            starttrial_code = '768'
            starttrial_events = np.array(events_type) == starttrial_code
            idxs = np.where(starttrial_events)[0]
            print(f"Subject {sub_id} - Start trial events: {len(idxs)} found")

            trial_labels = self.get_labels(sub_id)

            trials = []
            classes = []
            for j, index in enumerate(idxs):
                try:
                    # Append the label for the current trial
                    classes.append(trial_labels[j])

                    start = int(events_position[index])
                    stop = start + int(events_duration[index])

                    # Define the trial window based on your requirements
                    trial_start = int(start + 2 * self.Fs)  # Start at 2 seconds (2 * 250)
                    trial_stop = int(start + 5.5 * self.Fs)  # End at 5.5 seconds (5.5 * 250)

                    # Ensure valid interval
                    if trial_stop > raw.shape[1]:
                        print(f"Invalid trial interval for trial {j}: start={trial_start}, stop={trial_stop}, max={raw.shape[1]}")
                        continue

                    # Extract trial data ensuring correct dimensions
                    trial = raw[:22, trial_start:trial_stop]

                    # Check if the trial data has valid length
                    if trial.shape[1] != (5.5 - 2) * self.Fs:
                        print(f"Unexpected trial length for trial {j}, expected {(5.5 - 2) * self.Fs}, got {trial.shape[1]}")
                        continue

                    trials.append(trial)
                    # print(f"Loaded trial {j} from subject {sub_id} with shape {trial.shape}")
                except Exception as e:
                    print(f"Cannot load trial {j} from subject {sub_id}: {e}")
                    continue
            print(f"Total trials loaded from subject {sub_id}: {len(trials)}")
            return trials, classes
        except IndexError as e:
            print(f"IndexError for subject {sub_id}: {e}")
            print(f"Available data indices: {len(self.data_all)}")
            raise

    def get_labels(self, sub_id):
        label_path = os.path.join(self.root_path, "true_labels")
        base_name = os.path.basename(self.filenames[sub_id])
        sub_name = os.path.splitext(base_name)[0]
        label_file = os.path.join(label_path, sub_name + ".mat")
        
        print(f"Looking for label file: {label_file}")  # Debugging statement

        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        labels = loadmat(label_file)["classlabel"]
        return labels.squeeze() - 1

    def get_trials_all(self):
        trials_all = []
        labels_all = []
        total_num = []
        for sub_id in range(len(self.data_all)):
            trials, labels = self.get_trials_from_single_subj(sub_id)
            total_num.append(len(trials))
            
            trials_all.append(np.array(trials))
            labels_all.append(np.array(labels))

        print(f"Total number of trials: {total_num}")
        
        if trials_all:
            trials_all_arr = np.vstack(trials_all)
            trials_all_arr = self.map2pret(trials_all_arr)
            print(f"Number of trials after preprocessing: {trials_all_arr.shape[0]}")
            
            # Ensure that labels_all is a flattened array with a label for each trial
            labels_all_arr = np.concatenate(labels_all)
            print(f"Number of labels: {len(labels_all_arr)}")

            return self.normalize(trials_all_arr), labels_all_arr, total_num
        else:
            raise ValueError("No trial data available to concatenate")

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data, axis=1)  # Apply along the time axis
        
        return filtered_data

class EmotionDataset(EEGDataset):
    def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.data_all = []
        for subject_id in ([0,1,2,3,4,5,6,7,8,9]):
            pruned_path = str("/home/insane/Scrivania/eremus_npz/")
            sessions = getPrunedSessions(pruned_path)
            raw = mne.io.read_raw_eeglab(Path(pruned_path)/sessions[sub(subject_id)], verbose=False)
            data, _ = raw[:]
            data_dict = {
                    's': data,
                    'etyp': raw.annotations.description,
                    'epos': raw.annotations.onset * raw.info['sfreq'],
                    'edur': raw.annotations.duration * raw.info['sfreq'],
                    'artifacts': np.zeros_like(raw.annotations.onset)  # Modify as needed
            }
            self.data_all.append(data_dict)

        # Debugging: Check structure of data_all
        print(f"data_all structure: {[type(item) for item in self.data_all]}")
        # print(f"Example data structure: {self.data_all[0]}")

        
        # Load transformation matrix and ensure correct shape
        self.P = np.load("../inputs/tMatrix_value.npy")
        if self.P.shape != (22, 22):
            raise ValueError("Transformation matrix self.P must be 22x22.")

        self.trials, self.labels, self.num_trials_per_sub = self.get_trials_all()

    def __len__(self):
        total_len = sum(self.num_trials_per_sub)
        # print(f"Dataset length: {total_len}")
        return total_len

    def __getitem__(self, idx):
        # print(f"Accessing index: {idx} of {self.__len__()}")  # Debugging statement
        if idx >= len(self.trials) or idx < 0:
            raise IndexError(f"Index {idx} out of bounds for trials of size {len(self.trials)}")
        return self.preprocess_sample(self.trials[idx], self.num_chunks, self.labels[idx])

    def map2pret(self, data):
        # Apply transformation matrix to align channel configuration
        print(f"Shape of self.P: {self.P.shape}")  # Debugging statement
        print(f"Shape of data before transformation: {data.shape}")  # Debugging statement

        if data.shape[1] != 22:
            raise ValueError("Mismatch between data channels and transformation matrix channels.")

        # Perform transformation on (batch, channels, time) format
        transformed_data = np.einsum('ij,bjt->bit', self.P, data)
        print(f"Shape of data after transformation: {transformed_data.shape}")
        
        return transformed_data

    def get_trials_from_single_subj(self, sub_id):
        try:
            raw = self.data_all[sub_id]['s']  # (channels, time)
           
            # print(f"Subject {sub_id} annotations: {events_type}")
            self.xlsx = pd.read_excel("../augmented_eremus.xlsx")
            filter = self.xlsx["original_index"] == sub_id
            rows=self.xlsx.where(filter).dropna(thresh=1)
            idxs=[]
            for id in rows.iloc[:,0].index:
                idxs.append(id)
            emotions = rows.iloc[:,11]
            labels=[]
            for emotion in emotions: 
                labels.append(gew.gew_to_hldv4(eval(emotion)))
            trial_labels=np.array(labels).squeeze()
            
            trials = []
            classes = []
            for j, index in enumerate(idxs):
                # try:
                # Append the label for the current trial
                classes.append(trial_labels[j])
                
                trial_start=int(rows.iloc[j,6]) 
                trial_stop=trial_start+500
               
                # Ensure valid interval
                if trial_stop > raw.shape[1]:
                    print(f"Invalid trial interval for trial {j}: start={trial_start}, stop={trial_stop}, max={raw.shape[1]}")
                    continue

                # Extract trial data ensuring correct dimensions
                trial = raw[:22, trial_start:trial_stop]

                # Check if the trial data has valid length
                if trial.shape[1] != 500:
                    print(f"Unexpected trial length for trial {j}, expected {(5.5 - 2) * self.Fs}, got {trial.shape[1]}")
                    continue
                print(trial.shape)
                trials.append(trial)
               
            print(f"Total trials loaded from subject {sub_id}: {len(trials)}")
            
            return trials, classes

        except IndexError as e:
            print(f"IndexError for subject {sub_id}: {e}")
            print(f"Available data indices: {len(self.data_all)}")
            raise

    def get_labels(self, sub_id):
        label_path = os.path.join(self.root_path, "true_labels")
        base_name = os.path.basename(self.filenames[sub_id])
        sub_name = os.path.splitext(base_name)[0]
        label_file = os.path.join(label_path, sub_name + ".mat")
        
        print(f"Looking for label file: {label_file}")  # Debugging statement

        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        labels = loadmat(label_file)["classlabel"]
        return labels.squeeze() - 1

    def get_trials_all(self):
        trials_all = []
        labels_all = []
        total_num = []
        print(len(self.data_all))
        for sub_id in range(len(self.data_all)):
            trials, labels = self.get_trials_from_single_subj(sub_id)
            total_num.append(len(trials))
            
            trials_all.append(np.array(trials))
            labels_all.append(np.array(labels))

        print(f"Total number of trials: {total_num}")
        
        if trials_all:
            trials_all_arr = np.vstack(trials_all)
            trials_all_arr = self.map2pret(trials_all_arr)
            print(f"Number of trials after preprocessing: {trials_all_arr.shape[0]}")
            
            # Ensure that labels_all is a flattened array with a label for each trial
            labels_all_arr = np.concatenate(labels_all)
            print(f"Number of labels: {len(labels_all_arr)}")

            return self.normalize(trials_all_arr), labels_all_arr, total_num
        else:
            raise ValueError("No trial data available to concatenate")
