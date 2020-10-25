from Datasets.CV.CVDatasets import CIFAR10
db = CIFAR10()
from Datasets.BCI.SignalTarget import SignalAndTarget
from collections import OrderedDict
st = SignalAndTarget(db)
train_set,test_set = st.split_into_two_sets(0.7,shuffle=False)
train_set,valid_set = train_set.split_into_two_sets(0.7,shuffle=False)

from Estimators.Classifiers.ImageClassifiers.LeNet5 import LeNet5
"""
fit a model using conventional fit method

model_const=OrderedDict(img_size=(32,32),in_chan=3,n_classes=10)
model_hyper=OrderedDict(n_kernel_0=6,n_kernel_1=16,linear_init_std=1e-3,conv_init_bias=1e-3)

lm = LeNet5(model_const=model_const,model_hyper=model_hyper)
fit_const = {
    'train_set':train_set,
    'valid_set':valid_set,
    'test_set':test_set,
    'max_epochs':20,
    'max_increase_epochs':20,
    'cuda':True,
    'batch_size':128,
    'fitter':'VanillaFitter
}
optimizer_hyper = {
    'lr':1e-3,
}
loss_hyper = {
    'model_constraint':True
}
lm.compile(fit_const,optimizer_hyper,loss_hyper)
lm.fitter.run()
"""
from Estimators.NetworkTrainner.ModuleSelection.HyperParameter import RandomSearcher
model_hypers = OrderedDict({
    'linear_init_std':([0,0.5],'range','log'),
    'conv_init_bias':([0,1e-3],'range','linear'),
    'n_kernel_0':([5,6],'set',None),
    'n_kernel_1':([16,16],'set',None),
})
optimizer_hypers = OrderedDict({
    'lr':([1e-8,1e-7],'range','log')
})
loss_hypers = OrderedDict({
    'model_constraint':([False,True],'set',None)
})
model_const = {
    'img_size':(32,32),
    'in_chan':3,
    'n_classes':10,
}
fit_const = {
    'train_set':train_set,
    'valid_set':valid_set,
    'test_set':test_set,
    'max_epochs':20,
    'max_increase_epochs':20,
    'cuda':True,
    'batch_size':128,
    'fitter':'VanillaFitter'
}
RS = RandomSearcher(model_hypers=model_hypers,optimizer_hypers=optimizer_hypers,loss_hypers=loss_hypers
                  ,model_type=LeNet5,model_const=model_const,fit_const=fit_const)
RS.runKRandomSearchs(k=10,max_epochs=10)


# best_model = RS.best_model
# fit_const['max_epochs'] = 200
# fit_const['max_increase_epochs'] = 50
# best_model.compile(fit_const,RS.best_optim_hyper,RS.best_loss_hyper)
# best_model.fitter.run()
