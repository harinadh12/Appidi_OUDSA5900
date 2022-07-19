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
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from joblib import dump, load 
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# add 
# Tabulate Demographic charactersitics of participants
# 

LABEL_FREQUENCY = 700
DATASET_PATH = "../WESAD/"

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

    @staticmethod
    def get_wrist_frequencies():
        wrist_freq_dict = {}
        wrist_freq_dict['acc_wrist_freq'] = 32
        wrist_freq_dict['bvp_wrist_freq'] = 64
        wrist_freq_dict['eda_wrist_freq'] = 4
        wrist_freq_dict['temp_wrist_freq'] = 4

        return wrist_freq_dict

    # ['ACC', 'BVP', 'EDA', 'TEMP']
    def get_wrist_data(self):
        wrist_data = self.data.get('signal',{}).get('wrist',[])
        return wrist_data
    
    # ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
    def get_chest_data(self):
        chest_data = self.data.get('signal',{}).get('chest',[])
        return chest_data
    
    def get_labels(self):
        return self.data.get('label',None)
        
    def get_net_accel(self, data):
        return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))
    
    def aggregate_features(self, feature, frequency=700):
        mean_features = np.zeros(math.ceil(len(feature)/frequency))
        std_features = np.zeros(math.ceil(len(feature)/frequency))
        max_features = np.zeros(math.ceil(len(feature)/frequency))
        min_features = np.zeros(math.ceil(len(feature)/frequency))
        # sum_features = np.zeros(math.ceil(len(feature)/frequency))

        idx =0

        for i in range(0, len(feature),frequency):
            mean_features[idx] = np.mean(feature[i:i+frequency])
            std_features[idx] = np.std(feature[i:i+frequency])
            max_features[idx] = np.amax(feature[i:i+frequency])
            min_features[idx] = np.amin(feature[i:i+frequency])
            # sum_features[idx] = np.sum(feature[i:i+frequency])
            idx+=1
        
        return np.column_stack((max_features, min_features, std_features, mean_features))
        
    def extract_chest_features(self, chest_data):

        # chest_data = chest_data[indices]
        agg_data = np.apply_along_axis(self.aggregate_features, 0, chest_data)
        chest_data = agg_data.reshape(agg_data.shape[0],-1) # shape of (nrows, *4*n_columns)
        return chest_data

    def extract_wrist_features(self, wrist_data_dict):
        wrist_freq_dict = ReadSubjectData.get_wrist_frequencies()
        reshape = lambda x: x.reshape(x.shape[0],-1)
        agg_acc_wrist = reshape(np.apply_along_axis(self.aggregate_features,0,wrist_data_dict['ACC'],wrist_freq_dict['acc_wrist_freq']))
        agg_bvp_wrist = reshape(np.apply_along_axis(self.aggregate_features,0,wrist_data_dict['BVP'],wrist_freq_dict['bvp_wrist_freq']))
        agg_eda_wrist = reshape(np.apply_along_axis(self.aggregate_features,0,wrist_data_dict['EDA'],wrist_freq_dict['eda_wrist_freq']))
        agg_temp_wrist = reshape(np.apply_along_axis(self.aggregate_features,0,wrist_data_dict['TEMP'],wrist_freq_dict['temp_wrist_freq']))

        # print(agg_acc_wrist.shape, agg_bvp_wrist.shape, agg_eda_wrist.shape, agg_temp_wrist.shape)
        agg_wrist_data = np.concatenate([agg_acc_wrist,agg_bvp_wrist,agg_eda_wrist,agg_temp_wrist], axis=1)
        return agg_wrist_data
    
    def get_aggregate_labels(self):
        idx=0
        labels = self.get_labels()
        new_labels = np.empty((labels.shape[0]//LABEL_FREQUENCY),)

        for i in range(0, len(labels),LABEL_FREQUENCY):
            new_labels[idx] = np.rint(np.mean(labels[i:i+LABEL_FREQUENCY]))
            idx +=1

        return new_labels

    def pca_dimension_reduction(self, features):
        pca = PCA(n_components=8)
        features = pca.fit_transform(features)
        return features

    def standardize_data(self, features, train=True):
        
        if train is True:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            with open('standard_scaler.pkl','w') as f:
                dump(scaler,'scaler.gz')
        else:
            scaler = load('scaler.gz')
            features = scaler.transform(features)

        return features


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

def create_train_test_data(features, labels):

    dataset = {}
    dataset['features'] = features
    dataset['labels'] = labels



    dataset['features'], dataset['labels'] = shuffle(dataset['features'], dataset['labels'])

    train_dataset, test_dataset = collections.OrderedDict(), collections.OrderedDict()

    train_dataset['features'], test_dataset['features'], train_dataset['labels'], test_dataset['labels'] = train_test_split(dataset['features'],dataset['labels'], test_size=0.3, random_state=1, stratify=dataset['labels'])


    # train_dataset['features'] = obj.pca_dimension_reduction(train_dataset['features'])
    train_dataset['features'] = obj.standardize_data(train_dataset['features'] , train=True)

    # # test_dataset['features'] = obj.pca_dimension_reduction(test_dataset['features'])
    test_dataset['features'] = obj.standardize_data(test_dataset['features'] , train=False)

    return train_dataset, test_dataset

def create_aggregate_features(DATASET_PATH):
    pass

if __name__=="__main__":

    DATASET_PATH = "../WESAD/"
    pickle_files = get_pickle_files(dataset_path=DATASET_PATH)
    chest_data = {}
    chest_labels = {}
    # mean_data = 0


    for subject_no, file_path in pickle_files.items():
        if os.path.exists(f'{DATASET_PATH}/{subject_no}_train_data.npy')  :
            continue
        
        # create an object for read_subject_data passing pickle file
        obj = ReadSubjectData(
            file_path=file_path
        )

        # get chest data from the subject object 
        chest_data_dict = obj.get_chest_data()

        #get wrist data from the subject object
        wrist_data_dict = obj.get_wrist_data()

        # get labels(subject's activity) from subject object
        labels = obj.get_labels()
        keys = ['ACC','ECG','EMG','EDA','Temp','Resp']

        # extract feature from chest data di t and aggregate them based on frequency window
        chest_data = np.concatenate([chest_data_dict[key] for key in keys], axis=1)


        # Dimensions of different sensors used for chest data measurement
        # 'ACC' : 3, 'ECG' 1: , 'EDA' : 1, 'EMG': 1, 'RESP': 1, 'Temp': 1  ===> Total dimensions : 8
        
        # Labels --> 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
        # 4 = meditation, 5/6/7 = should be ignored in this dataset

        # Let's slice chest data for only labels --> 0,1,2,3,4 and ignore other labels in this experiment

        
        chest_features = obj.extract_chest_features(chest_data=chest_data)
        
        wrist_features = obj.extract_wrist_features(wrist_data_dict)
        
        
        labels = obj.get_aggregate_labels()
        
        chest_features = chest_features[np.where(np.logical_and(labels>0, labels<4))[0]]
        
        wrist_features = wrist_features[np.where(np.logical_and(labels>0, labels<4))[0]]
        
        features = np.concatenate([chest_features,wrist_features], axis=1)
        labels = labels[np.where(np.logical_and(labels>0, labels<4))[0]]

        train_dataset, test_dataset = create_train_test_data(features=features, labels=labels)
        
        with open(f'{DATASET_PATH}/{subject_no}_train_data.npy','wb') as f:
            np.save(f, train_dataset)
        
        with open(f'{DATASET_PATH}/{subject_no}_test_data.npy','wb') as f:
            np.save(f, test_dataset)