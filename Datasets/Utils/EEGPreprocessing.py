import mne
import numpy as np
from Datasets.Utils.SignalProcessing import BandpassCnt,HighpassCnt,LowpassCnt,ExponentialRunningStandardize


def RawApply(func, raw, verbose="WARNING"):
    """
    Apply function to data of `mne.io.RawArray`.

    Parameters
    ----------
    func: function
        Should accept 2d-array (channels x time) and return modified 2d-array
    raw: `mne.io.RawArray`
    verbose: bool
        Whether to log creation of new `mne.io.RawArray`.

    Returns
    -------
    transformed_set: Copy of `raw` with data transformed by given function.

    """
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=verbose)

def EpochsApply(func, epochs):
    """Apply function to data of 'mne.Epochs'"""
    new_data = epochs.get_data()
    for i_epoch,epoch in enumerate(epochs):
        new_data[i_epoch] = func(epoch)
    return mne.EpochsArray(new_data,epochs.info,epochs.events,epochs.tmin,epochs.event_id)

def ToRawData(epochs_data):
    """
    convert 3d array to 2d in Epochs fashion
    :param epochs_data:(n_trials,n_chan,n_timesteps)
    :return: raw_data
    """
    return np.concatenate([e for e in epochs_data],axis=1)



def ReplaceRawNan(raw,verbose='WARNING'):
    data = raw.get_data()
    for i_chan,chan_data in enumerate(data):
        this_chan = chan_data
        #To soften the effect of very small numbers when computing average of chan_data
        chan_data = np.where(this_chan==np.min(this_chan),np.nan,this_chan)
        mask = np.isnan(chan_data)
        chan_mean = np.nanmean(chan_data)
        chan_data[mask] = chan_mean
    return mne.io.RawArray(data, raw.info, verbose=verbose)

def ReplaceEpochsNan(epochs):
    epochs_data = epochs.get_data()
    for raw in epochs_data:
        for i_chan, chan_data in enumerate(raw):
            this_chan = chan_data
            # To soften the effect of very small numbers when computing average of chan_data
            chan_data = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
            mask = np.isnan(chan_data)
            chan_mean = np.nanmean(chan_data)
            chan_data[mask] = chan_mean
    return mne.EpochsArray(epochs_data, epochs.info, epochs.events, epochs.tmin, epochs.event_id)

def ReplaceDatasetNan(dataset,raw_version,modified_version):
    for i_subject,EEG_subject in enumerate(dataset.subjects_data):
        raw = EEG_subject.subject_cnt[raw_version]
        if type(raw)==mne.io.RawArray:
            new = ReplaceRawNan(raw)
        elif type(raw)==mne.EpochsArray:
            new = ReplaceEpochsNan(raw)
        EEG_subject.subject_cnt[modified_version] = new



def RawUv(raw):
    return RawApply(lambda a:a*1e6, raw)

def EpochsUv(epochs):
    return EpochsApply(lambda a:a*1e6, epochs)

def DatasetUv(dataset,raw_version,modified_version):
    for i_subject,EEG_subject in enumerate(dataset.subjects_data):
        raw = EEG_subject.subject_cnt[raw_version]
        if type(raw)==mne.io.RawArray:
            new = RawUv(raw)
        elif type(raw)==mne.EpochsArray:
            new = EpochsUv(raw)
        EEG_subject.subject_cnt[modified_version] = new



def NormalizeRaw(raw):
    return RawApply(lambda a:(a-np.min(a)/(np.max(a)-np.min(a))),raw)

def NormalizeEpochs(epochs):
    return EpochsApply(lambda a:(a-np.min(a)/(np.max(a)-np.min(a))),epochs)

def NormalizeDataset(dataset,raw_version,modified_version):
    for i_subject,EEG_subject in enumerate(dataset.subjects_data):
        raw = EEG_subject.subject_cnt[raw_version]
        if type(raw)==mne.io.RawArray:
            new = NormalizeRaw(raw)
        elif type(raw)==mne.EpochsArray:
            new = NormalizeEpochs(raw)
        EEG_subject.subject_cnt[modified_version] = new



def StandardizationRaw(raw):
    return RawApply(lambda a:(a-np.mean(a,axis=0)/(np.std(a,axis=0))),raw)

def StandardizationEpochs(epochs):
    return EpochsApply(lambda a:(a-np.mean(a,axis=0)/(np.std(a,axis=0))),epochs)

def StandardizationDataset(dataset,raw_version,modified_version):
    for i_subject,EEG_subject in enumerate(dataset.subjects_data):
        raw = EEG_subject.subject_cnt[raw_version]
        if type(raw)==mne.io.RawArray:
            new = StandardizationRaw(raw)
        elif type(raw)==mne.EpochsArray:
            new = StandardizationEpochs(raw)
        EEG_subject.subject_cnt[modified_version] = new

def RawBandPass(raw,low_cut_hz,high_cut_hz,fs,filt_order=3,axis=1):
    return RawApply(lambda a:BandpassCnt(a,low_cut_hz=low_cut_hz,
                                         high_cut_hz=high_cut_hz,
                                         fs=fs,
                                         filt_order=filt_order,
                                         axis=axis),raw)

def EpochsBandPass(epochs,low_cut_hz,high_cut_hz,fs,filt_order=3,axis=1):
    return EpochsApply(lambda a:BandpassCnt(a,low_cut_hz=low_cut_hz,
                                         high_cut_hz=high_cut_hz,
                                         fs=fs,
                                         filt_order=filt_order,
                                         axis=axis),epochs)

def DatasetBandPass(dataset,raw_version,modified_version,low_cut_hz,
                    high_cut_hz,fs,filt_order=3,axis=1):
    for i_subject,EEG_subject in enumerate(dataset.subjects_data):
        raw = EEG_subject.subject_cnt[raw_version]
        if type(raw)==mne.io.RawArray:
            new = RawBandPass(raw,low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz,fs=fs,filt_order=filt_order,axis=axis)
        elif type(raw)==mne.EpochsArray:
            new = EpochsBandPass(raw,low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz,fs=fs,filt_order=filt_order,axis=axis)
        EEG_subject.subject_cnt[modified_version] = new



def RawExpRunStandard(raw,factor_new=0.001,eps=1e-5):
    return RawApply(lambda a:ExponentialRunningStandardize(a,factor_new=factor_new,eps=eps),raw)

def EpochsExpRunStandard(epochs,factor_new=0.001,eps=1e-5):
    return EpochsApply(lambda a:ExponentialRunningStandardize(a,factor_new=factor_new,eps=eps),epochs)

def DatasetExpRunStandard(dataset,raw_version,modified_version,factor_new=0.001,eps=1e-5):
    for i_subject,EEG_subject in enumerate(dataset.subjects_data):
        raw = EEG_subject.subject_cnt[raw_version]
        if type(raw)==mne.io.RawArray:
            new = RawExpRunStandard(raw,factor_new=factor_new,eps=eps)
        elif type(raw)==mne.EpochsArray:
            new = EpochsExpRunStandard(raw,factor_new=factor_new,eps=eps)
        EEG_subject.subject_cnt[modified_version] = new

