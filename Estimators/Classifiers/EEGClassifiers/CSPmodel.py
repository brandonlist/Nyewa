import numpy as np
from sklearn import svm
from mne.decoding import CSP
import sklearn.feature_selection
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from Datasets.Utils.SignalProcessing import BandpassCnt
from Estimators.NetworkTrainner.Fitters.InstantFitters import InstantFitter

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
    band pass and return features filtered by csp
    :param model:
    :param freq:
    :param data:
    :return:
    """
    features = []
    for freq_count, lower in enumerate(model.freq[:-1]):
        # loop for freqency
        higher = model.freq[freq_count + 1]
        x_filt = BandpassCnt(x, low_cut_hz=lower, high_cut_hz=higher, fs=model.fs, filt_order=model.filt_order, axis=2)
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

def create_features_using_csp(model,x,y=None):
    if type(y) == type(None):
        # using transform method
        feature = model.csp.transform(x)
    else:
        # using fit method
        feature = model.csp.fit_transform(x, y)
    return feature

class CSPBase(BaseEstimator,ClassifierMixin):
    """
    filter the eeg data in given frequency range
    select specific time window of the data
    learn csp features
    make classification using SVC
    """
    def __init__(self,model_const,model_hyper):
        self.model_const = model_const
        self.model_hyper = model_hyper

        self.fs = model_const['fs']
        self.time_steps = model_const['time_steps']
        self.low_cut_hz = model_hyper['low_cut_hz']
        self.high_cut_hz = model_hyper['high_cut_hz']

        self.window_start = model_hyper['window_start']
        self.window_length = model_hyper['window_length']
        self.n_csp_component = model_hyper['n_csp_component']
        self.filt_order = model_hyper['filt_order']

        self.time_window = create_time_window(window_start=self.window_start,
                                              window_length=self.window_length,
                                              time_steps=self.time_steps)
        self.clf = svm.SVC(C=0.8, kernel='rbf')
        self.csp = CSP(n_components=self.n_csp_component, reg=None, log=True, norm_trace=False)

    def fit(self,x,y):
        """
        The following parameters are defined:
        ys, clf ,csp
        :param x: [n_trials, n_chan, time_steps]
        :param y:
        :return:
        """
        self.ys = np.unique(y)
        x_filt = BandpassCnt(x, low_cut_hz=self.low_cut_hz, high_cut_hz=self.high_cut_hz,
                             fs=self.fs, filt_order=self.filt_order, axis=2)
        x_filt = x_filt[:, :, self.time_window[0]:self.time_window[1]]
        feature = create_features_using_csp(self,x_filt,y)
        self.clf.fit(feature,y)

    def transform(self,x):
        x_filt = BandpassCnt(x, low_cut_hz=self.low_cut_hz, high_cut_hz=self.high_cut_hz,
                             fs=self.fs, filt_order=self.filt_order, axis=2)
        x_filt = x_filt[:, :, self.time_window[0]:self.time_window[1]]
        feature = create_features_using_csp(self, x_filt)
        return feature

    def predict(self,x):
        features = self.transform(x)
        pred = self.clf.predict(features)
        return pred

    def score(self,X,y,sample_weight=None):
        acc = accuracy_score(y,self.predict(X))
        return acc

    def compile(self,fit_const,optimizer_hyper,loss_hyper):
        train_set = fit_const['train_set']
        valid_set = fit_const['valid_set']
        test_set = fit_const['test_set']
        fitter = fit_const['fitter']

        if fitter=='InstantFitter':
            self.fitter = InstantFitter(
                self,
                train_set,
                valid_set,
                test_set
            )

class CSSP(BaseEstimator,ClassifierMixin):
    """
    filter the eeg data in given frequency range
    stack T-delayed window data below
    select specific time window of the data
    learn csp features of the stacked data
    make classification using SVC
    """
    def __init__(self,model_const,model_hyper):
        self.model_const = model_const
        self.model_hyper = model_hyper

        self.T = model_hyper['T']
        self.fs = model_const['fs']
        self.time_steps = model_const['time_steps']
        self.low_cut_hz = model_hyper['low_cut_hz']
        self.high_cut_hz = model_hyper['high_cut_hz']

        self.window_start = model_hyper['window_start']
        self.window_length = model_hyper['window_length']
        self.n_csp_component = model_hyper['n_csp_component']
        self.filt_order = model_hyper['filt_order']

        self.time_window = create_time_window(window_start=self.window_start,
                                              window_length=self.window_length,
                                              time_steps=self.time_steps)
        self.clf = svm.SVC(C=0.8, kernel='rbf')
        self.csp = CSP(n_components=self.n_csp_component, reg=None, log=True, norm_trace=False)

    def fit(self,x,y):
        x_filt = BandpassCnt(x, low_cut_hz=self.low_cut_hz, high_cut_hz=self.high_cut_hz,
                             fs=self.fs, filt_order=self.filt_order, axis=2)
        x_filt = np.hstack([x_filt[..., self.T:], x_filt[..., :-self.T]])
        x_filt = x_filt[:, :, self.time_window[0]:self.time_window[1]]
        feature = create_features_using_csp(self,x_filt,y)
        self.clf.fit(feature,y)

    def transform(self,x):
        x_filt = BandpassCnt(x, low_cut_hz=self.low_cut_hz, high_cut_hz=self.high_cut_hz,
                             fs=self.fs, filt_order=self.filt_order, axis=2)
        x_filt = np.hstack([x_filt[..., self.T:], x_filt[..., :-self.T]])
        x_filt = x_filt[:, :, self.time_window[0]:self.time_window[1]]
        feature = create_features_using_csp(self, x_filt)
        return feature

    def predict(self,x):
        features = self.transform(x)
        pred = self.clf.predict(features)
        return pred

    def score(self, X, y, sample_weight=None):
        acc = accuracy_score(y, self.predict(X))
        return acc

    def compile(self, fit_const, optimizer_hyper, loss_hyper):
        train_set = fit_const['train_set']
        valid_set = fit_const['valid_set']
        test_set = fit_const['test_set']
        fitter = fit_const['fitter']

        if fitter == 'InstantFitter':
            self.fitter = InstantFitter(
                self,
                train_set,
                valid_set,
                test_set
            )

class FBCSP(BaseEstimator,ClassifierMixin):
    """
    select specific time window of the data
    filter eeg data into different sub-bands,
        learn csp for each bands then using this csp for transform stage
    select features using mutual information
    make classification using SVC
    """
    def __init__(self,model_const,model_hyper):
        self.model_const = model_const
        self.model_hyper = model_hyper

        self.fs = model_const['fs']
        self.time_steps = model_const['time_steps']

        self.low_cut_hz = model_hyper['low_cut_hz']
        self.cut_inter = model_hyper['cut_inter']
        self.n_cuts = model_hyper['n_cuts']
        self.select_ratio = model_hyper['select_ratio']
        self.window_start = model_hyper['window_start']
        self.window_length = model_hyper['window_length']
        self.n_csp_component = model_hyper['n_csp_component']
        self.filt_order = model_hyper['filt_order']

        self.csps = []
        self.clf = svm.SVC(C=0.8, kernel='rbf')
        self.freq = create_filter_band(low_cut_hz=self.low_cut_hz,
                                       cut_inter=self.cut_inter,
                                       n_cuts=self.n_cuts)
        self.time_window = create_time_window(window_start=self.window_start,
                                              window_length=self.window_length,
                                              time_steps=self.time_steps)
        self.csps = []

    def fit(self,x,y):
        """

        :param x:
        :param y:
        The following parameters are defined:
        ys, k , clf ,ss , select_K , csps
        elements of csps and freq should be matched
        :return:
        """
        self.ys = np.unique(y)
        x = x[:,:,self.time_window[0]:self.time_window[1]]
        features = band_pass_trials_using_filter_bank(self,x,y)
        #features :[n_trials,n_features]
        self.k = int(features.shape[1] * self.select_ratio)
        self.select_K = sklearn.feature_selection.SelectKBest(mutual_info_classif, k=self.k)
        self.select_K.fit(features,y)
        features = self.select_K.transform(features)
        self.ss = preprocessing.StandardScaler()
        features = self.ss.fit_transform(features,y)
        self.clf.fit(features,y)

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

        if fitter=='InstantFitter':
            self.fitter = InstantFitter(
                self,
                train_set,
                valid_set,
                test_set
            )

