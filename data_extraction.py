import pickle
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pathlib
import shutil
import math
from scipy.stats import mode
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle
# add 
# Tabulate Demographic charactersitics of participants
# 


class ReadSubjectData:
    def __init__(self, file_path):
        """ Read Subject data from pkl file"""
        
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')

        self.data = data
        self.subject_no = self.data.get('subject',None)
        # self.labels = self.data['label']
    
    #['label', 'subject', 'signal']
    def get_data_keys(self):
        return self.data.keys()

    #['wrist', 'chest']
    def get_signal_types(self):
        return self.data.get('signal',{}).keys()

    #['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
    def get_wrist_data(self):
        wrist_data = self.data.get('signal',{}).get('wrist',[])
        return wrist_data
    
    # ['ACC', 'BVP', 'EDA', 'TEMP']
    def get_chest_data(self):
        chest_data = self.data.get('signal',{}).get('chest',[])
        return chest_data
    
    def get_labels(self):
        return self.data.get('label',None)
        
    def get_net_accel(self, data):
        return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))
    
    def aggreage_features(self, feature, frequency=1400):
        mean_features = np.zeros(math.ceil(len(feature)/frequency))
        std_features = np.zeros(math.ceil(len(feature)/frequency))
        max_features = np.zeros(math.ceil(len(feature)/frequency))
        min_features = np.zeros(math.ceil(len(feature)/frequency))
        sum_features = np.zeros(math.ceil(len(feature)/frequency))

        idx =0

        for i in range(0, len(feature),frequency):
            mean_features[idx] = np.mean(feature[i:i+frequency])
            std_features[idx] = np.std(feature[i:i+frequency])
            max_features[idx] = np.amax(feature[i:i+frequency])
            min_features[idx] = np.amin(feature[i:i+frequency])
            sum_features[idx] = np.sum(feature[i:i+frequency])
            idx+=1
        
        return np.column_stack((max_features, min_features, std_features, mean_features, sum_features))
        
    def extract_chest_features(self, chest_data, indices):

        chest_data = chest_data[indices]
        agg_data = np.apply_along_axis(self.aggreage_features, 0, chest_data)
        chest_data = agg_data.reshape(agg_data.shape[0],-1) # shape of (nrows, *4*n_columns)
        return chest_data

def get_pickle_files(dataset_path):
    """
    """
    pickle_files = {}
    for root, dirnames, filenames in os.walk(dataset_path, topdown=True):
        for file_name in filenames :
            if file_name.endswith('.pkl'):
                subject_no = file_name.split('.')[0]
                pickle_files[subject_no] = os.path.join(root,file_name)

    return pickle_files

def create_aggregate_features(DATASET_PATH):

    DATASET_PATH = "./WESAD/"
    pickle_files = get_pickle_files(dataset_path=DATASET_PATH)
    chest_data = {}
    chest_labels = {}
    for subject_no, file_path in pickle_files.items():

        if os.path.exists(f'{DATASET_PATH}/{subject_no}_data.npy')  :
            continue
        
        print(file_path)

        # create an object for read_subject_data passing pickle file
        obj = read_subject_data(
            file_path=file_path
        )

        # get chest data from the subject object 
        chest_data_dict = obj.get_chest_data()

        # get labels(subject's activity) from subject object
        chest_label = obj.get_labels()

        # extract feature from chest data di t and aggregate them based on frequency window
        chest_data = np.concatenate([chest_data_dict[key] for key in chest_data_dict], axis=1)

        # Dimensions of different sensors used for chest data measurement
        # 'ACC' : 3, 'ECG' 1: , 'EDA' : 1, 'EMG': 1, 'RESP': 1, 'Temp': 1  ===> Total dimensions : 8

        # Labels --> 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
        # 4 = meditation, 5/6/7 = should be ignored in this dataset

        # Let's slice chest data for only labels --> 0,1,2,3,4 and ignore other labels in this experiment
        chest_features = []
        chest_labels = []
        
        for i in range(1,5):
            # label_indices = np.where(np.logical_and(chest_label>0, chest_label<4))
            label_indices = np.where(chest_label==i)
            features = obj.extract_chest_features(chest_data=chest_data, indices=label_indices)
            chest_features.append(features)
            chest_labels.extend([i]*features.shape[0])
        
        
        dataset = {}
        dataset['features'] = np.vstack(chest_features)
        dataset['labels'] = np.asarray(chest_labels)
        dataset['features'], dataset['labels'] = shuffle(dataset['features'], dataset['labels'])

        train_dataset, test_dataset = collections.OrderedDict(), collections.OrderedDict()

        train_dataset['features'], test_dataset['features'], train_dataset['labels'], test_dataset['labels'] = train_test_split(dataset['features'],dataset['labels'], test_size=0.3, random_state=1)

        with open(f'{DATASET_PATH}/{subject_no}_train_data.npy','wb') as f:
            np.save(f, train_dataset)
        
        with open(f'{DATASET_PATH}/{subject_no}_test_data.npy','wb') as f:
            np.save(f, test_dataset)

if __name__=="__main__":

    DATASET_PATH = "./WESAD/"
    pickle_files = get_pickle_files(dataset_path=DATASET_PATH)
    chest_data = {}
    chest_labels = {}
    for subject_no, file_path in pickle_files.items():

        if os.path.exists(f'{DATASET_PATH}/{subject_no}_train_data.npy')  :
            continue
        
        print(file_path)

        # create an object for read_subject_data passing pickle file
        obj = ReadSubjectData(
            file_path=file_path
        )

        # get chest data from the subject object 
        chest_data_dict = obj.get_chest_data()

        # get labels(subject's activity) from subject object
        chest_label = obj.get_labels()

        # extract feature from chest data di t and aggregate them based on frequency window
        chest_data = np.concatenate([chest_data_dict[key] for key in chest_data_dict], axis=1)

        # Dimensions of different sensors used for chest data measurement
        # 'ACC' : 3, 'ECG' 1: , 'EDA' : 1, 'EMG': 1, 'RESP': 1, 'Temp': 1  ===> Total dimensions : 8
        # Labels --> 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
        # 4 = meditation, 5/6/7 = should be ignored in this dataset

        # Let's slice chest data for only labels --> 0,1,2,3,4 and ignore other labels in this experiment
        chest_features = []
        chest_labels = []
        
        for i in range(1,4):
            # label_indices = np.where(np.logical_and(chest_label>0, chest_label<4))
            label_indices = np.where(chest_label==i)
            features = obj.extract_chest_features(chest_data=chest_data, indices=label_indices)
            chest_features.append(features)
            chest_labels.extend([i]*features.shape[0])
        
        
        dataset = {}
        dataset['features'] = np.vstack(chest_features)
        dataset['labels'] = np.asarray(chest_labels)
        dataset['features'], dataset['labels'] = shuffle(dataset['features'], dataset['labels'])

        train_dataset, test_dataset = collections.OrderedDict(), collections.OrderedDict()

        train_dataset['features'], test_dataset['features'], train_dataset['labels'], test_dataset['labels'] = train_test_split(dataset['features'],dataset['labels'], test_size=0.3, random_state=1)

        with open(f'{DATASET_PATH}/{subject_no}_train_data.npy','wb') as f:
            np.save(f, train_dataset)
        
        with open(f'{DATASET_PATH}/{subject_no}_test_data.npy','wb') as f:
            np.save(f, test_dataset)