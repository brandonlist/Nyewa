from Datasets.BCI.BCIDataBase import EEGDatabase
from Datasets.Utils.EEGPreprocessing import DatasetBandPass,DatasetUv,ReplaceDatasetNan,DatasetExpRunStandard,\
    NormalizeDataset,StandardizationDataset
from scipy.io import loadmat
import mne
import os
import numpy as np
from collections import OrderedDict

base_dir = r'H:\Datasets\MIDatabase'

def preprocess_standard(dataset,raw_version,modified_version='standard',low_cut_hz=4,high_cut_hz=38):
    #TODO:交换顺序与选择都有不便，需要修改，或者把这个当作固定的函数，另写一个便于修改的函数
    DatasetUv(dataset, raw_version=raw_version, modified_version='temp')
    DatasetExpRunStandard(dataset, raw_version='temp', modified_version='temp')
    DatasetBandPass(dataset, raw_version='temp', modified_version='temp', low_cut_hz=low_cut_hz,
                    high_cut_hz=high_cut_hz, fs=dataset.fs)
    ReplaceDatasetNan(dataset, raw_version='temp', modified_version=modified_version)
    #in case there is only one change
    try:
        for EEG_subject in dataset.subjects_data:
            del EEG_subject.subject_cnt['temp']
    except:
        return

def get_raw_trial_from_gdf_file(dataset, filename, classes=None ):
    raw_t = mne.io.read_raw_gdf(filename, stim_channel='auto')
    data = raw_t.get_data()
    gdf_events = mne.events_from_annotations(raw_t)
    info = mne.create_info(dataset.channel_names, dataset.fs, dataset.channel_types)
    info.set_montage(dataset.montage)
    raw = mne.io.RawArray(data, info, verbose="WARNING")
    raw.info["gdf_events"] = gdf_events
    for a_t in raw_t.annotations:
        a = OrderedDict()
        for key in a_t:
            a[key] = a_t[key]
        raw.annotations.append(onset=a['onset'], duration=a['duration'], description=a['description'])

    # extract events
    events, name_to_code = raw.info["gdf_events"]
    # name_to_code:{'1023': 1,'1072': 2,'276': 3,'277': 4,'32766': 5, '768': 6,'769': 7,'770': 8,'771': 9,'772': 10}
    if dataset.name == 'BCIC4_2a':
        # There is file in that dataset that lacks 2 EOG signal
        if len(name_to_code) == 8:  # The one file that lacks 2 EOG signal
            trial_codes = [5, 6, 7, 8]
            num_to_detract = 5
            code_start = 4
        elif '783' in name_to_code.keys():  # Reading label file
            trial_codes = [7]
            num_to_detract = 0
            code_start = 6
        else: #normal occasion for that dataset
            trial_codes = [name_to_code[d] for d in dataset.eventDescription if 'Onset' in dataset.eventDescription[d]]
    else:
        trial_codes = [name_to_code[d] for d in dataset.eventDescription if 'Onset' in dataset.eventDescription[d]]

    num = trial_codes
    num.sort()
    num_to_detract = num[0]
    del num
    code_start = name_to_code['768']

    trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
    trial_events = events[trial_mask]

    trial_events[:, 2] = trial_events[:, 2] - num_to_detract

    if dataset.name == 'BCIC4_2a':
        # cause only read label from that dataset now
        if '783' in name_to_code.keys():  # meaning is test file
            trial_events[:, 2] = classes
            classes_set = [7, 8, 9, 10]

    raw.info['events'] = trial_events
    # unique_classes = np.unique(trial_events[:, 2])

    if dataset.name=='BCIC3_3b':
        #no rejected trials in that dataset file
        return raw

    # now also create 0-1 vector for rejected trials
    if dataset.name == 'BCIC4_2a' or dataset.name == 'BCIC4_2b':
        trial_start_events = events[events[:, 2] == code_start]
    if dataset.name == 'BCIC3_3a':
        events_no1 = events[events[:, 2] != 1]
        t_time = [events_no1[i + 3, 0] for i, c in enumerate(events_no1[:-3, 2]) if
                  c == 2 and (events_no1[i + 3, 2] in [3, 4, 5, 6])]
        mask = [(t in t_time) for t in events[:,0]]
        trial_start_events = events[mask]
    assert len(trial_start_events)==len(trial_events),print(len(trial_start_events),len(trial_events))
    artifact_trial_mask = np.zeros(len(trial_start_events), dtype=np.uint8)
    artifact_events = events[events[:, 2] == name_to_code['1023']]
    for artifact_time in artifact_events[:, 0]:
        try:
            i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
        except:
            continue
        artifact_trial_mask[i_trial] = 1

    # get trial events and artifact_mask

    raw.info['artifact_trial_mask'] = artifact_trial_mask

    # drop EOG channels
    if dataset.name == 'BCIC4_2a':
        raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

    # for events in raw, there are only 4 task events 0,1,2,3
    return raw

class BCIC2_3(EEGDatabase):
    def __init__(self,path=os.path.join(base_dir,r'BCIC2\3\dataset_BCIcomp1.mat')):
        super(BCIC2_3, self).__init__(name='BCIC2_3',n_subject=1)
        self.path = path
        self.fs = 128
        self.n_classes = 2
        self.channel_names = ['C3','Cz','C4']
        self.channel_types = ['eeg'] * 3
        self.montage = 'standard_1005'
        self.ys = (1,2) #class type that was recorded in the original data file
        self.chi_names = ['左手','右手']

        #construct data structure
        data_all = loadmat(self.path)
        x_data = data_all['x_train']
        y_data = data_all['y_train']
        x_data = np.swapaxes(x_data, 0, 2)
        # x_data:[140(trials),3(channels),1152(time_steps)]
        super(BCIC2_3, self).init_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            info = mne.create_info(self.channel_names, self.fs, self.channel_types)
            info.set_montage(self.montage)
            events = np.zeros((140,3),dtype=np.int)
            events[:, 2] = y_data.squeeze()
            events[:, 0] = np.arange(140).transpose()
            event_id = dict(ImaginedLeftHand=1,ImaginedRightHand=2)
            tmin = 0
            raw = mne.EpochsArray(x_data,info,events,tmin,event_id)

            self.subjects_data[i].subject_cnt = {}
            self.subjects_data[i].subject_cnt['raw_epochs'] = raw

    def preprocess_standard(self,low_cut_hz=4,high_cut_hz=38):
        preprocess_standard(self,raw_version='raw_epochs',low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz)

    def load_data(self,version='raw_epochs'):
        for i in range(self.n_subject):
            self.subjects_data[i].load_data(n_trial=140)
            for idx,trial in enumerate(self.subjects_data[i].subject_cnt[version]):
                self.subjects_data[i].subject_trials[idx].signal = trial
                self.subjects_data[i].subject_trials[idx].target = self.subjects_data[i].subject_cnt[version].events[idx,2]

class BCIC3_3a(EEGDatabase):
    def __init__(self,path=os.path.join(base_dir,r'BCIC3\3\3a')):
        super(BCIC3_3a, self).__init__(name='BCIC3_3a',n_subject=3)
        self.path = path
        self.fs = 250
        self.n_classes = 4
        self.channel_names = ['AFFz','F1h','Fz','F2h','FFC1','FFC1h','FFCz','FFC2h','FFC2','FC3h','FC1','FC1h','FCz',
                              'FC2h','FC2','FC4h','FCC3','FCC3h','FCC1','FCC1h','FCCz','FCC2h','FCC2','FCC4h','FCC4',
                              'C5','C5h','C3','C3h','C1h','Cz','C2h','C4h','C4','C6h','C6','CCP3','CCP3h','CCP1','CCP1h',
                              'CCPz','CCP2h','CCP2','CCP4h','CCP4','CP3h','CP1','CP1h','CPz','CP2h','CP2','CP4h','CPP1',
                              'CPP1h','CPPz','CPP2h','CPP2','P1h','Pz','P2h']
        self.eventDescription = {'768': 'startTrail',
                                 '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight',
                                 '783': 'cueUnknown',
                                 '1023': 'rejectedTrial',
                                 '785': 'beep',
                                 '786': 'crossOnScreen'}
        self.names = ['#  1', '#  2', '#  3', '#  4', '#  5', '#  6', '#  7', '#  8', '#  9', '# 10', '# 11', '# 12', '# 13', '# 14',
                     '# 15', '# 16', '# 17', '# 18', '# 19', '# 20', '# 21', '# 22', '# 23', '# 24', '# 25', '# 26', '# 27', '# 28',
                     '# 29', '# 30', '# 31', '# 32', '# 33', '# 34', '# 35', '# 36', '# 37', '# 38', '# 39', '# 40', '# 41', '# 42',
                     '# 43', '# 44', '# 45', '# 46', '# 47', '# 48', '# 49', '# 50', '# 51', '# 52', '# 53', '# 54', '# 55', '# 56',
                     '# 57', '# 58', '# 59', '# 60']
        self.channel_types = ['eeg'] * 60
        self.montage = 'standard_1005'
        self.chi_names = ['左手', '右手', '脚', '舌头']
        self.C3 = 28;self.C4 = 34

        #construct data structure
        super(BCIC3_3a, self).init_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            path_s = os.path.join(self.path, 's' + str(i + 1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf':
                    path_s = os.path.join(path_s, file_path)
                    raw = get_raw_trial_from_gdf_file(self,path_s)
                    self.subjects_data[i].subject_cnt = {}
                    self.subjects_data[i].subject_cnt['raw_cnt'] = raw

    def preprocess_standard(self,low_cut_hz=4,high_cut_hz=38):
        preprocess_standard(self,raw_version='raw_cnt',low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz)

    def load_data(self,version='raw_cnt'):
        for i in range(self.n_subject):
            raw = self.subjects_data[i].subject_cnt[version]
            n_trial = len(raw.info['events'])
            self.subjects_data[i].load_data(n_trial=n_trial)

            events = raw.info['events']
            event_id = dict(ImaginedLeftHand=0,ImaginedRightHand=1)
            epochs = mne.Epochs(raw, events, event_id, event_repeated='merge', tmin=-0.5, tmax=4)
            for idx,trial in enumerate(epochs):
                self.subjects_data[i].subject_trials[idx].signal = trial
                self.subjects_data[i].subject_trials[idx].target = raw.info['events'][idx,2]



class BCIC3_3b(EEGDatabase):
    def __init__(self,path=os.path.join(base_dir,r'BCIC3\3\3b')):
        super(BCIC3_3b, self).__init__(name='BCIC3_3b',n_subject=3)
        self.path = path
        self.fs = 125
        self.n_classes = 2
        self.channel_names = ['C3','C4']
        self.eventDescription = {'768': 'startTrail',
                                 '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight',
                                 '781':'feedbackOnsetContinuous',
                                 '783': 'cueUnknown',
                                 '785': 'beep'}
        self.names = ['+C3a-C3p','+C4a-C4p']
        self.channel_types = ['eeg'] * 2
        self.montage = 'standard_1005'
        self.chi_names = ['左手','右手']
        self.C3 = 0;self.C4 = 1

        super(BCIC3_3b, self).init_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            path_s = os.path.join(self.path, 's' + str(i + 1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf':
                    path_s = os.path.join(path_s, file_path)
                    raw = get_raw_trial_from_gdf_file(self, path_s)
                    self.subjects_data[i].subject_cnt = {}
                    self.subjects_data[i].subject_cnt['raw_cnt'] = raw

    def preprocess_standard(self,low_cut_hz=4,high_cut_hz=38):
        preprocess_standard(self,raw_version='raw_cnt',low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz)

    def load_data(self,version='raw_cnt'):
        for i in range(self.n_subject):
            raw = self.subjects_data[i].subject_cnt[version]
            n_trial = len(raw.info['events'])
            self.subjects_data[i].load_data(n_trial=n_trial)

            events = raw.info['events']
            event_id = dict(ImaginedLeftHand=0,ImaginedRightHand=1)
            epochs = mne.Epochs(raw, events, event_id, event_repeated='merge', tmin=-0.5, tmax=4)
            for idx,trial in enumerate(epochs):
                self.subjects_data[i].subject_trials[idx].signal = trial
                self.subjects_data[i].subject_trials[idx].target = raw.info['events'][idx,2]

class BCIC3_4a(EEGDatabase):
    def __init__(self,path=os.path.join(base_dir,r'BCIC3\4\4a\1000Hz')):
        super(BCIC3_4a, self).__init__(name='BCIC3_4a',n_subject=5)
        self.path = path
        self.fs = 1000
        self.n_classes = 2
        self.channel_names = ['Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'AFF5', 'AFF1', 'AFF2', 'AFF6','F7',
                      'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFT7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4','FFC6',
                      'FFT8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'FTT7','FCC5',
                      'FCC3', 'FCC1', 'FCC2', 'FCC4', 'FCC6', 'FTT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6','T8',
                      'TTP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'TTP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'TPP7', 'CPP5', 'CPP3', 'CPP1', 'CPP2', 'CPP4', 'CPP6',
                      'TPP8', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PPO7', 'PPO5', 'PPO1',
                      'PPO2', 'PPO6', 'PPO8', 'PO7', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO8', 'POO1', 'POO2', 'O1','Oz',
                      'O2', 'OI1', 'OI2', 'I1', 'I2']
        self.names = ['Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'FAF5', 'FAF1', 'FAF2', 'FAF6','F7',
                      'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFC7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4','FFC6',
                      'FFC8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'CFC7','CFC5',
                      'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6','T8',
                      'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP7', 'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6',
                      'PCP8', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PPO7', 'PPO5', 'PPO1',
                      'PPO2', 'PPO6', 'PPO8', 'PO7', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO8', 'OPO1', 'OPO2', 'O1','Oz',
                      'O2', 'OI1', 'OI2', 'I1', 'I2']
        self.channel_types = ['eeg'] * 118
        self.montage = 'standard_1005'
        self.ys = (1, 2)  # 指原始文件里记录的类型值
        self.chi_names = ['右手', '脚']
        self.tmin = 0
        self.tmax = 4
        self.pre_steps = int(self.tmin*self.fs)
        self.post_steps = int(self.tmax*self.fs)
        self.n_trials = []
        # construct data structure
        super(BCIC3_4a, self).init_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            info = mne.create_info(self.channel_names,self.fs,self.channel_types)
            info.set_montage(self.montage)

            file_mat_path = [p for p in os.listdir(os.path.join(self.path, 's' + str(i + 1))) if p.split('.')[-1] == 'mat'][0]
            file_mat_path = os.path.join(self.path, 's' + str(i + 1), file_mat_path)
            data_subject_i = loadmat(file_mat_path)

            mrk_y = data_subject_i['mrk']['y'].item()
            mrk_pos = data_subject_i['mrk']['pos'].item()
            mask_all = (mrk_y < 5)[0]
            n_trial = mask_all.sum();self.n_trials.append(n_trial)
            y = mrk_y[mrk_y<5]
            index_list = mrk_pos[0,mask_all]
            data_i = []
            for trial_i in range(n_trial):
                signal = data_subject_i['cnt'][index_list[trial_i]-self.pre_steps:index_list[trial_i]+self.post_steps,:]
                signal = np.swapaxes(signal,0,1)
                data_i.append(signal)
            data_i = np.array(data_i)

            events = np.zeros((n_trial,3),dtype=np.int)
            events[:,2] = y
            events[:,0] = np.arange(n_trial).transpose()
            event_id = dict(ImaginedRightHand=1,ImaginedFeet=2)
            raw = mne.EpochsArray(data_i, info, events, self.tmin, event_id)

            self.subjects_data[i].subject_cnt = {}
            self.subjects_data[i].subject_cnt['raw_epochs'] = raw

    def preprocess_standard(self,low_cut_hz=4,high_cut_hz=38):
        preprocess_standard(self,raw_version='raw_epochs',low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz)

    def load_data(self,version='raw_epochs'):
        for i in range(self.n_subject):
            self.subjects_data[i].load_data(n_trial=self.n_trials[i])
            for idx,trial in enumerate(self.subjects_data[i].subject_cnt[version]):
                self.subjects_data[i].subject_trials[idx].signal = trial
                self.subjects_data[i].subject_trials[idx].target = self.subjects_data[i].subject_cnt[version].events[idx,2]

class BCIC3_4b(EEGDatabase):
    def __init__(self,path=os.path.join(base_dir,r'BCIC3\4\4b\1000Hz')):
        super(BCIC3_4b, self).__init__(name='BCIC3_4b',n_subject=1)
        self.path = path
        self.fs = 1000
        self.n_classes = 2
        self.chi_names = ['左手','脚']
        self.channel_names = ['Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'AFF5', 'AFF1', 'AFF2', 'AFF6','F7',
                      'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFT7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4','FFC6',
                      'FFT8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'FTT7','FCC5',
                      'FCC3', 'FCC1', 'FCC2', 'FCC4', 'FCC6', 'FTT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6','T8',
                      'TTP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'TTP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'TPP7', 'CPP5', 'CPP3', 'CPP1', 'CPP2', 'CPP4', 'CPP6',
                      'TPP8', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PPO7', 'PPO5', 'PPO1',
                      'PPO2', 'PPO6', 'PPO8', 'PO7', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO8', 'POO1', 'POO2', 'O1','Oz',
                      'O2', 'OI1', 'OI2', 'I1', 'I2']
        self.names = ['Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'FAF5', 'FAF1', 'FAF2', 'FAF6','F7',
                      'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFC7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4','FFC6',
                      'FFC8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'CFC7','CFC5',
                      'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6','T8',
                      'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP7', 'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6',
                      'PCP8', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PPO7', 'PPO5', 'PPO1',
                      'PPO2', 'PPO6', 'PPO8', 'PO7', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO8', 'OPO1', 'OPO2', 'O1','Oz',
                      'O2', 'OI1', 'OI2', 'I1', 'I2']
        self.channel_types = ['eeg'] * 118
        self.montage = 'standard_1005'
        self.ys = (-1, 1)  # 指原始文件里记录的类型值
        self.tmin = 0
        self.tmax = 4
        self.pre_steps = int(self.tmin * self.fs)
        self.post_steps = int(self.tmax * self.fs)
        self.n_trials = []
        # construct data structure
        super(BCIC3_4b, self).init_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            info = mne.create_info(self.channel_names, self.fs, self.channel_types)
            info.set_montage(self.montage)

            file_mat_path = os.path.join(self.path,'data_set_IVb_al_train.mat')
            data_subject_i = loadmat(file_mat_path)

            mrk_y = data_subject_i['mrk']['y'].item()
            mrk_pos = data_subject_i['mrk']['pos'].item()
            mask_all = (mrk_y < 5)[0]
            n_trial = mask_all.sum();
            self.n_trials.append(n_trial)
            y = mrk_y[mrk_y < 5]
            index_list = mrk_pos[0, mask_all]
            data_i = []
            for trial_i in range(n_trial):
                signal = data_subject_i['cnt'][
                         index_list[trial_i] - self.pre_steps:index_list[trial_i] + self.post_steps, :]
                signal = np.swapaxes(signal, 0, 1)
                data_i.append(signal)
            data_i = np.array(data_i)

            events = np.zeros((n_trial, 3), dtype=np.int)
            events[:, 2] = y
            events[:, 0] = np.arange(n_trial).transpose()
            event_id = dict(ImaginedLeftHand=-1, ImaginedFeet=1)
            raw = mne.EpochsArray(data_i, info, events, self.tmin, event_id)

            self.subjects_data[i].subject_cnt = {}
            self.subjects_data[i].subject_cnt['raw_epochs'] = raw

    def preprocess_standard(self,low_cut_hz=4,high_cut_hz=38):
        preprocess_standard(self,raw_version='raw_epochs',low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz)

    def load_data(self,version='raw_epochs'):
        for i in range(self.n_subject):
            self.subjects_data[i].load_data(n_trial=self.n_trials[i])
            for idx,trial in enumerate(self.subjects_data[i].subject_cnt[version]):
                self.subjects_data[i].subject_trials[idx].signal = trial
                self.subjects_data[i].subject_trials[idx].target = self.subjects_data[i].subject_cnt[version].events[idx,2]

class BCIC4_1(EEGDatabase):
    # TODO：每个被试的类别实际上都不一样，在info[classes]里面
    def __init__(self,path=os.path.join(base_dir,r'BCIC4\1\1_1000Hz')):
        super(BCIC4_1, self).__init__(name='BCIC4_1',n_subject=7)
        self.path = path
        self.fs = 1000
        self.n_classes = 2
        self.chi_names = ['左手','右手']
        self.channel_names = ['AF3','AF4','F5','F3','F1','Fz','F2','F4','F6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FTT7',
                    'FCC5','FCC3','FCC1','FCC2','FCC4','FCC6','FTT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','TTP7',
                      'CCP5','CCP3','CCP1','CCP2','CCP4','CCP6','TTP8','CP5','CP3','CP1','CPz','CP2','CP4','CP6','P5',
                      'P3','P1','Pz','P2','P4','P6','PO1','PO2','O1', 'O2']
        self.names = ['AF3','AF4','F5','F3','F1','Fz','F2','F4','F6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CFC7',
                    'CFC5','CFC3','CFC1','CFC2','CFC4','CFC6','CFC8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','CCP7',
                      'CCP5','CCP3','CCP1','CCP2','CCP4','CCP6','CCP8','CP5','CP3','CP1','CPz','CP2','CP4','CP6','P5',
                      'P3','P1','Pz','P2','P4','P6','PO1','PO2','O1', 'O2']
        self.channel_types = ['eeg'] * 59
        self.montage = 'standard_1005'
        self.ys = (-1, 1)  # 指原始文件里记录的类型值
        self.tmin = 0
        self.tmax = 4
        self.pre_steps = int(self.tmin * self.fs)
        self.post_steps = int(self.tmax * self.fs)
        self.n_trials = []
        # construct data structure
        super(BCIC4_1, self).init_data()
        file_names = [p for p in os.listdir(self.path) if p.split('.')[-1] == 'mat']
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            info = mne.create_info(self.channel_names, self.fs, self.channel_types)
            info.set_montage(self.montage)

            file_mat_path = os.path.join(self.path,file_names[i])
            data_subject_i = loadmat(file_mat_path)

            mrk_y = data_subject_i['mrk']['y'].item()
            mrk_pos = data_subject_i['mrk']['pos'].item()
            mask_all = (mrk_y < 5)[0]
            n_trial = mask_all.sum();
            self.n_trials.append(n_trial)
            y = mrk_y[mrk_y < 5]
            index_list = mrk_pos[0, mask_all]
            data_i = []
            for trial_i in range(n_trial):
                signal = data_subject_i['cnt'][
                         index_list[trial_i] - self.pre_steps:index_list[trial_i] + self.post_steps, :]
                signal = np.swapaxes(signal, 0, 1)
                data_i.append(signal)
            data_i = np.array(data_i)

            events = np.zeros((n_trial, 3), dtype=np.int)
            events[:, 2] = y
            events[:, 0] = np.arange(n_trial).transpose()
            event_id = dict(ImaginedLeftHand=-1, ImaginedRightHand=1)
            raw = mne.EpochsArray(data_i, info, events, self.tmin, event_id)

            self.subjects_data[i].subject_cnt = {}
            self.subjects_data[i].subject_cnt['raw_epochs'] = raw

    def preprocess_standard(self,low_cut_hz=4,high_cut_hz=38):
        preprocess_standard(self,raw_version='raw_epochs',low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz)

    def load_data(self, version='raw_epochs'):
        for i in range(self.n_subject):
            self.subjects_data[i].load_data(n_trial=self.n_trials[i])
            for idx, trial in enumerate(self.subjects_data[i].subject_cnt[version]):
                self.subjects_data[i].subject_trials[idx].signal = trial
                self.subjects_data[i].subject_trials[idx].target = self.subjects_data[i].subject_cnt[version].events[
                    idx, 2]

class BCIC4_2a(EEGDatabase):
    def __init__(self,path=os.path.join(base_dir,r'BCIC4\2\a')):
        super(BCIC4_2a, self).__init__(name='BCIC4_2a',n_subject=9)
        self.path = path
        self.fs = 250
        self.n_classes = 4
        self.channel_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                      'C2', 'C4', 'C6',
                      'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz', 'EOG-left', 'EOG-central', 'EOG-right']
        self.channel_types = ['eeg'] * 22 + ['eog'] * 3
        self.montage = 'standard_1005'
        self.chi_names = ['左手', '右手', '脚', '舌头']
        self.eventDescription = {'276': 'eyesOpen',
                                 '277': 'eyesClosed',
                                 '768': 'startTrail',
                                 '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight',
                                 '771': 'cueOnsetFoot',
                                 '772': 'cueOnsetTongue',
                                 '783': 'cueUnknown',
                                 '1023': 'rejectedTrial',
                                 '1072': 'eyeMovements',
                                 '32766': 'startOfNewRun'}
        self.eog_names = ['EOG-left',
                          'EOG-central',
                          'EOG-right']
        self.names = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz',
                      'EEG-7', 'EEG-C4', 'EEG-8',
                      'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']
        self.C3 = 7;self.C4 = 11

        # construct data structure
        super(BCIC4_2a, self).init_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            path_s = os.path.join(self.path, 's' + str(i + 1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf' and file_path.find('T') != -1:
                    path_s_t = os.path.join(path_s, file_path)
                if file_path.split('.')[-1] == 'gdf' and file_path.find('E') != -1:
                    path_s_e = os.path.join(path_s, file_path)
                if file_path.split('.')[-1] == 'mat' and file_path.find('E') != -1:
                    path_label = os.path.join(path_s, file_path)
            classes = loadmat(path_label)["classlabel"].squeeze() + 6
            raw_t = get_raw_trial_from_gdf_file(self,path_s_t)
            raw_e = get_raw_trial_from_gdf_file(self,path_s_e,classes)
            # raw_t = self._get_raw_from_filename(path_s_t)
            # raw_e = self._get_raw_from_filename(path_s_e,classes)
            raw = mne.concatenate_raws([raw_t,raw_e])
            self.subjects_data[i].subject_cnt = {}
            self.subjects_data[i].subject_cnt['raw_cnt'] = raw

    def preprocess_standard(self,low_cut_hz=4,high_cut_hz=38):
        preprocess_standard(self,raw_version='raw_cnt',low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz)

    def load_data(self,version='raw_cnt'):
        for i in range(self.n_subject):
            raw = self.subjects_data[i].subject_cnt[version]
            n_trial = len(raw.info['events'])
            self.subjects_data[i].load_data(n_trial=n_trial)

            events = raw.info['events']
            event_id = dict(ImaginedLeftHand=0,ImaginedRightHand=1,ImaginedFeet=2,ImaginedTongue=3)
            epochs = mne.Epochs(raw, events, event_id, event_repeated='merge', tmin=-0.5, tmax=4)
            for idx,trial in enumerate(epochs):
                self.subjects_data[i].subject_trials[idx].signal = trial
                self.subjects_data[i].subject_trials[idx].target = raw.info['events'][idx,2]

    #original version of function get_raw_from_filename, dont use any more
    def _get_raw_from_filename(self,filename,classes=None):
        raw_t = mne.io.read_raw_gdf(filename,stim_channel='auto')
        data = raw_t.get_data()
        gdf_events = mne.events_from_annotations(raw_t)
        info = mne.create_info(self.channel_names, self.fs, self.channel_types)
        info.set_montage(self.montage)
        raw = mne.io.RawArray(data, info, verbose="WARNING")
        raw.info["gdf_events"] = gdf_events
        for a_t in raw_t.annotations:
            a = OrderedDict()
            for key in a_t:
                a[key] = a_t[key]
            raw.annotations.append(onset=a['onset'],duration=a['duration'],description=a['description'])

        #extract events
        events, name_to_code = raw.info["gdf_events"]
        #name_to_code:{'1023': 1,'1072': 2,'276': 3,'277': 4,'32766': 5, '768': 6,'769': 7,'770': 8,'771': 9,'772': 10}
        trial_codes = [7,8,9,10]
        num_to_detract = 7
        code_start = 6
        if len(name_to_code) == 8: #The one file that lacks 2 EOG signal
            trial_codes = [5,6,7,8]
            num_to_detract = 5
            code_start = 4
        if '783' in name_to_code.keys():#Reading label file
            trial_codes = [7]
            num_to_detract = 0
            code_start = 6
        trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
        trial_events = events[trial_mask]
        assert len(trial_events) == 288, "Got {:d} markers".format(
            len(trial_events)
        )
        #4 classes marked with 0,1,2,3
        trial_events[:, 2] = trial_events[:, 2] - num_to_detract
        classes_set = [0,1,2,3]
        if '783' in name_to_code.keys(): #meaning is test file
            trial_events[:,2] = classes
            classes_set = [7,8,9,10]

        unique_classes = np.unique(trial_events[:, 2])
        assert np.array_equal(
            classes_set, unique_classes
        ), "Expect 0,1,2,3 or 7,8,9,10 as class labels, got {:s}".format(
            str(unique_classes)
        )

        # now also create 0-1 vector for rejected trials
        trial_start_events = events[events[:, 2] == code_start]
        assert len(trial_start_events) == len(trial_events)
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        artifact_events = events[events[:, 2] == 1]

        for artifact_time in artifact_events[:, 0]:
            i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
            artifact_trial_mask[i_trial] = 1

        #get trial events and artifact_mask
        raw.info['events'] = trial_events
        raw.info['artifact_trial_mask'] = artifact_trial_mask

        #drop EOG channels
        raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

        #for events in raw, there are only 4 task events 0,1,2,3
        return raw

class BCIC4_2b(EEGDatabase):
    def __init__(self,path=os.path.join(base_dir,r'BCIC4\2\b')):
        super(BCIC4_2b, self).__init__(name='BCIC4_2b',n_subject=9)
        self.path = path
        self.fs = 250
        self.n_classes = 2
        self.chi_names = ['左手','右手']
        self.channel_names = ['C3','Cz','C4','EOG:01','EOG:02','EOG:03']
        self.names = ['EEG:C3','EEG:Cz','EEG:C4','EOG:ch01','EOG:ch02','EOG:ch03']
        self.eventDescription = {'768': 'startTrail',
                                 '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight',
                                 '783': 'cueUnknown',
                                 '1023': 'rejectedTrial',
                                 '1077':'horizontalEyeMovement',
                                 '1078':'verticalEyeMovement',
                                 '1079':'eyeRotationClockwise',
                                 '1081':'eyeBlinks'}
        self.channel_types = ['eeg'] * 3 + ['eog'] * 3
        self.montage = 'standard_1005'
        self.C3 = 0;self.C4 = 2

        # construct data structure
        super(BCIC4_2b, self).init_data()
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            path_s = os.path.join(self.path,'s'+str(i+1))
            self.path_s_ts = []
            self.raws = []
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf' and file_path.find('T') !=-1:
                    path_s_t = os.path.join(path_s,file_path)
                    self.path_s_ts.append(path_s_t)
                if file_path.split('.')[-1] == 'gdf' and file_path.find('E') !=-1:
                    path_s_e = os.path.join(path_s,file_path)
            for file_path in self.path_s_ts:
                raw = get_raw_trial_from_gdf_file(self,file_path)
                self.raws.append(raw)
            #TODO:add raw_e to raw
            raw = mne.concatenate_raws([r for r in self.raws])
            self.subjects_data[i].subject_cnt = {}
            self.subjects_data[i].subject_cnt['raw_cnt'] = raw

    def preprocess_standard(self,low_cut_hz=4,high_cut_hz=38):
        preprocess_standard(self,raw_version='raw_cnt',low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz)

    def load_data(self,version='raw_cnt'):
        for i in range(self.n_subject):
            raw = self.subjects_data[i].subject_cnt[version]
            n_trial = len(raw.info['events'])
            self.subjects_data[i].load_data(n_trial=n_trial)

            events = raw.info['events']
            event_id = dict(ImaginedLeftHand=0,ImaginedRightHand=1)
            epochs = mne.Epochs(raw, events, event_id, event_repeated='merge', tmin=-0.5, tmax=4)
            for idx,trial in enumerate(epochs):
                self.subjects_data[i].subject_trials[idx].signal = trial
                self.subjects_data[i].subject_trials[idx].target = raw.info['events'][idx,2]



# db = BCIC4_2a()
# db.load_data()
# from Datasets.Utils.SignalAndTarget import SignalAndTarget
# st = SignalAndTarget(db,[0,1])
# p_st = st.apply_to_X_y(lambda a:a[:50])
# t,e = st.split_into_two_sets(0.7,shuffle=True)
# from Datasets.Utils.SignalAndTarget import concatenate_sets
# ans = concatenate_sets([t,e])