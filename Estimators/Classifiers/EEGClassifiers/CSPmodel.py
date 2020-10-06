import numpy as np
from sklearn import svm
from mne.decoding import CSP
import sklearn.feature_selection
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from Datasets.Utils.SignalProcessing import BandpassCnt
from Estimators.NetworkTrainner.Fitters.BaseFitters import InstantBaseFitter

random.seed(1)

from Datasets.BCI.BCIDatasets import BCIC2_3
from Datasets.BCI.SignalTarget import SignalAndTarget
db = BCIC2_3()
db.preprocess_standard(high_cut_hz=32)
db.load_data()

signal_and_target = SignalAndTarget(dataset=db)
train_set,test_set = signal_and_target.split_into_two_sets(n_first_set=110,shuffle=True)
X_train,X_test = train_set.X,test_set.X
y_train,y_test = train_set.y,test_set.y



model_const = {
    'fs':db.fs,
    'time_steps':1152
}

model_hyper = {
    'low_cut_hz':4,
    'cut_inter':4,
    'n_cuts':6,
    'select_ratio':5/6,
    'window_start':448,
    'window_length':448,
    'n_csp_component':2
}

def create_filter_band(low_cut_hz,cut_inter,n_cuts):
    """

    :param low_cut_hz: <int>
    :param cut_inter: <int>
    :param n_cuts: <int>
    :return:<list>filter_bank or <None>
    """
    low_cut_hz,cut_inter,n_cuts = int(low_cut_hz),int(cut_inter),int(n_cuts)
    if n_cuts<=0 or low_cut_hz<=0 or cut_inter<=0:
        return None
    filter_bank = [low_cut_hz]
    for i in range(n_cuts):
        filter_bank.append((i+1)*cut_inter+low_cut_hz)
    return filter_bank

def create_time_window(window_start,window_length,time_steps):
    """

    :param window_start:<int>
    :param window_length: <int>
    :param time_steps: <int>
    :return: <list>time_window
    """
    window_start,window_length = int(window_start),int(window_length)
    end = window_start+window_length if (window_start+window_length)<time_steps else time_steps
    return window_start,end

def band_pass_trials_using_filter_bank(model,x,y=None):
    """

    :param model:
    :param freq:
    :param data:
    :return:
    """
    features = []
    for freq_count, lower in enumerate(model.freq[:-1]):
        # loop for freqency
        higher = model.freq[freq_count + 1]
        x_filt = BandpassCnt(x, low_cut_hz=lower, high_cut_hz=higher, fs=model.fs, filt_order=8, axis=2)
        if type(y)==type(None):
            #using transform method
            tmp = model.csps[freq_count].transform(x_filt)
        else:
            #using fit method
            csp = CSP(n_components=model.n_csp_component, reg=None, log=True, norm_trace=False)
            tmp = csp.fit_transform(x_filt, y)
            model.csps.append(csp)
        if freq_count == 0:
            features = tmp
        else:
            features = np.concatenate((features, tmp), axis=1)
    return features

class FBCSP(BaseEstimator,ClassifierMixin):
    def __init__(self,model_const,model_hyper):
        self.fs = model_const['fs']
        self.time_steps = model_const['time_steps']

        self.low_cut_hz = model_hyper['low_cut_hz']
        self.cut_inter = model_hyper['cut_inter']
        self.n_cuts = model_hyper['n_cuts']
        self.select_ratio = model_hyper['select_ratio']
        self.window_start = model_hyper['window_start']
        self.window_length = model_hyper['window_length']
        self.n_csp_component = model_hyper['n_csp_component']

        self.csps = []
        self.clf = svm.SVC(C=0.8, kernel='rbf')

    def fit(self,x,y):
        """

        :param x:
        :param y:
        The following parameters are defined:
        ys, freq, time_window, k , clf ,ss , select_K , csps
        elements of csps and freq should be matched
        :return:
        """
        self.ys = np.unique(y)
        self.freq = create_filter_band(low_cut_hz=self.low_cut_hz,
                                       cut_inter=self.cut_inter,
                                       n_cuts=self.n_cuts)
        self.csps = []
        self.time_window = create_time_window(window_start=self.window_start,
                                              window_length=self.window_length,
                                              time_steps=self.time_steps)
        x = x[:,:,self.time_window[0]:self.time_window[1]]
        features = band_pass_trials_using_filter_bank(self,x,y)
        self.tp = features
        #features :[n_trials,n_features]
        self.k = int(features.shape[1] * self.select_ratio)
        self.select_K = sklearn.feature_selection.SelectKBest(mutual_info_classif, k=self.k)
        self.select_K.fit(features,y)
        features = self.select_K.transform(features)
        self.ss = preprocessing.StandardScaler()
        features = self.ss.fit_transform(features,y)
        self.clf.fit(features,y)
        return features

    def predict(self,x):
        """

        :param x: raw data
        :return:
        """
        features = self.transform(x)
        pred = self.clf.predict(features)
        return pred

    def transform(self,x):
        """

        :param x: raw data
        :return: features of trials
        """
        x = x[:, :, self.time_window[0]:self.time_window[1]]
        features = band_pass_trials_using_filter_bank(self,x)
        features = self.select_K.transform(features)
        features = self.ss.transform(features)
        return features

    def score(self,X,y,sample_weight=None):
        acc = accuracy_score(y,self.predict(X))
        return acc

    def compile(self,fit_const,optimizer_hyper,loss_hyper):
        train_set = fit_const['train_set']
        valid_set = fit_const['valid_set']
        test_set = fit_const['test_set']
        fitter = fit_const['fitter']

        if fitter=='InstantBaseFitter':
            self.fitter = InstantBaseFitter(
                self,
                train_set,
                valid_set,
                test_set
            )

model = FBCSP(model_const=model_const,model_hyper=model_hyper)
model.fit(X_train,y_train)
ans = model.score(X_test,y_test)

# In[4]:

#
# csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
#
# # In[5]:
#
# # acquire and combine features of different fequency bands
# features_train = []
# features_test = []
# freq = [8, 12, 16, 20, 24, 28, 32]
# for freq_count,lower in enumerate(freq[:-1]):
#     # loop for freqency
#     higher = freq[freq_count + 1]
#     X_train_filt = BandpassCnt(X_train, low_cut_hz=lower, high_cut_hz=higher, fs=db.fs, filt_order=8,axis=2)
#     X_test_filt = BandpassCnt(X_test, low_cut_hz=lower, high_cut_hz=higher, fs=db.fs, filt_order=8,axis=2)
#     tmp_train = csp.fit_transform(X_train_filt, y_train)
#     tmp_test = csp.transform(X_test_filt)
#     if freq_count == 0:
#         features_train = tmp_train
#         features_test = tmp_test
#     else:
#         features_train = np.concatenate((features_train, tmp_train), axis=1)
#         features_test = np.concatenate((features_test, tmp_test), axis=1)
# print(features_train.shape)  #(110,12)
# print(features_test.shape)  #(30,12)
#
# # In[6]:
#
# # get the best k features base on MIBIF algorithm
# select_K = sklearn.feature_selection.SelectKBest(mutual_info_classif, k=10).fit(features_train, y_train)
# New_train = select_K.transform(features_train)
# # np.random.shuffle(New_train)
# New_test = select_K.transform(features_test)
# # np.random.shuffle(New_test)
# print(New_train.shape) #(110,10)
# print(New_test.shape)  #(30,10)
# ss = preprocessing.StandardScaler()
# # X_select_train = New_train
# # X_select_test = New_test
# X_select_train = ss.fit_transform(New_train, y_train)
# X_select_test = ss.fit_transform(New_test)
#
# # In[7]:
#
#
# # calssify
# from sklearn.svm import SVC
#
# clf = svm.SVC(C=0.8, kernel='rbf')
# clf.fit(X_select_train, y_train)
# y_pred = clf.predict(X_select_test)
# print(y_test) #[0 1 1 1 1 0 0 1 1 0 0 1 0 1 1 1 1 0 0 0 0 1 0 1 0 1 1 1 1 0]
# print(y_pred) #[0 0 0 1 1 0 0 1 1 0 0 1 0 1 0 1 1 0 0 0 1 1 0 1 0 1 1 1 1 0]
# acc = accuracy_score(y_test, y_pred)
# print(acc)