import copy
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

class Trial(object):
    """
    Totally abstract object
    attributes can be added as needed
    We recommend the following attribute names to be identical when reading different datasets:
        signal
        target
    for continues data, might contain multiple labels in one trial
    """
    pass

class EEGDataSubject(object):
    """
    EEG data for one subject
    expected updating to succeed self-define general form of data
    """
    def __init__(self,id):
        self.id = id

    def load_data(self,n_trial):
        "parasing different form of data"
        self.n_trial = n_trial
        self.subject_trials = []
        for i in range(self.n_trial):
            trial = Trial()
            self.subject_trials.append(trial)

class EEGDatabase(Dataset):
    """
    General EEG data baseform, designed for multiple general-purpose use of the data, including:
    1.creat machine learning dataset SignalAndTarget
    2.analysis of the data including visualization
    3....
    """
    def __init__(self,name,n_subject):
        super(EEGDatabase, self).__init__()
        self.n_subject = n_subject
        self.name = name
        self.data_loaded = False
        self.type = 'BCI'

    def init_data(self):
        print("loading datasets: {0} ".format(self.name))
        self.subjects_data = []
        for i in range(self.n_subject):
            subject = EEGDataSubject(id=i)
            self.subjects_data.append(subject)

    def __len__(self):
        count = 0
        for subject in self.subjects_data:
            for _ in subject.subject_trials:
                count = count+1
        return count

    def __getitem__(self, idx):
        count = 0
        for subject_i,subject_data in enumerate(self.subjects_data):
            for trial_i,trial_data in enumerate(subject_data.subject_trials):
                if count == idx:
                    return trial_data.signal,trial_data.target
                count = count+1

    def memory(self):
        print(sys.getsizeof(self) / 1024 / 1024, 'MB')


