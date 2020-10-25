from Datasets.BCI.BCIDatasets import *
from Datasets.BCI.SignalTarget import SignalAndTarget
from Estimators.Classifiers.EEGClassifiers.CSPmodel import CSPBase
import random





db = BCIC3_3a()
db.preprocess_standard()
db.load_data()

signal_and_target = SignalAndTarget(dataset=db,subject_ids=[0])
random.seed(2)
train_set,test_set = signal_and_target.split_into_two_sets(n_first_set=80,shuffle=True)


model_const = {
    'fs':db.fs,
    'time_steps':db.time_steps
}
fit_const = {
    'train_set':train_set,
    'valid_set':None,
    'test_set':test_set,
    'fitter':'InstantFitter'
}

#suppose we hava find the best parameter for the dataset
model_hyper = {
    'low_cut_hz':4,
    'high_cut_hz':38,
    'window_start':448,
    'window_length':448,
    'n_csp_component':2,
    'filt_order':2
}


model = CSPBase(model_const=model_const,model_hyper=model_hyper)
model.compile(fit_const=fit_const,optimizer_hyper=None,loss_hyper=None)
ans = model.fitter.run()

