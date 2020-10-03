import numpy as np
import random
from collections import OrderedDict
import time
import matplotlib.pyplot as plt
import copy

# def get_random_param(params,range=False,log=None):
#     """
#     return random value of set in params
#     :param params:
#     for range==False, params can be:
#         test_params=OrderedDict({
#         'p1':[1,2,3,4],
#         'p2':[1,3,4,5],
#         'p3':np.arange(5)
#     })
#     for range==True, params can be:
#         test_params_range=OrderedDict({
#             'p1':[1,4],
#             'p2':[1.2,5.6],
#             'p3':[1e-5,1e-2],
#             'p4':[True,False]
#         })
#     :param range: Should be set to True if only the min and max are provided, Otherwise False
#     :param log: Whether to use logspace, can only be used when range is True
#     :return:
#     """
#     #TODO:range can be list indicating which param is using range
#     epsilon = 1e-10
#     if log==None:
#         log = [False] * len(params)
#     if range==False:
#         values = {}
#         for i,key in enumerate(params):
#             assert np.array(range).any()==False,print('can only use log space when parameter range is True')
#             idx = random.randint(0, len(params[key])-1)
#             value = params[key][idx]
#             values[key] = value
#         return values
#     elif range==True:
#         values = {}
#         for i,key in enumerate(params):
#             assert len(params[key])==2,print('please enter 2 nums if using range')
#             if log[i]==False:
#                 if type(params[key][0])==bool:
#                     value = random.uniform(0,1) < 0.5
#                 else:
#                     assert params[key][0] < params[key][1], print('first value have to be smaller than the latter one')
#                     value = random.uniform(float(params[key][0]),float(params[key][1]))
#                     if type(params[key][0])==int:
#                         value = int(value)
#                 values[key] = value
#             elif log[i]==True:
#                 low_bound = np.array(params[key][0])
#                 high_bound = np.array(params[key][1])
#                 value = random.uniform(np.log10(low_bound+epsilon),np.log10(high_bound+epsilon))
#                 value = np.power(10,value)
#                 values[key] = value
#         return values

model_hypers = OrderedDict({
    'linear_init_std':([0,0.5],'range','log'),
    'conv_init_bias':([0,1e-3],'range','linear'),
    'n_kernel_0':([5,6],'set',None),
    'n_kernel_1':([15,16],'set',None),
})

def _get_random_param(params):
    """

    """
    epsilon = 1e-10
    values = {}
    for i,key in enumerate(params):
        items = params[key]
        para,range,log = items[0],items[1],items[2]
        if range=='range':
            assert len(para)==2,print('please enter 2 nums if using range')
            if log=='linear':
                if type(para[0]) == bool:
                    value = random.uniform(0, 1) < 0.5
                else:
                    value = random.uniform(float(para[0]), float(para[1]))
                    if type(para[0]) == int:
                        value = int(value)
            elif log=='log':
                low_bound = np.array(para[0])
                high_bound = np.array(para[1])
                value = random.uniform(np.log10(low_bound+epsilon),np.log10(high_bound+epsilon))
                value = np.power(10,value)
            else:
                print('unknown type of log')
                return None
        elif range=='set':
            assert log==None,print('must set log to None if using set')
            idx = random.randint(0, len(para)-1)
            value = para[idx]
        else:
            print('unknown type of range')
            return None
        values[key] = value
    return values


class RandomSearcher(object):
    """
    model should have method model.fit

    """
    def __init__(self,model_hypers,optimizer_hypers,loss_hypers,model_type,
                 model_const,fit_const):
        self.model_hypers = model_hypers
        self.optimizer_hypers = optimizer_hypers
        self.loss_hypers = loss_hypers
        self.model_const = model_const
        self.fit_const = fit_const

        self.model_type = model_type
        self.models = []


    def getModelValue(self):
        model_hyper = _get_random_param(self.model_hypers)
        return model_hyper

    def getOptimValue(self):
        optimizer_hyper = _get_random_param(self.optimizer_hypers)
        return optimizer_hyper

    def getLossValue(self):
        loss_hyper = _get_random_param(self.loss_hypers)
        return loss_hyper

    def update_hyper(self):
        self.model_hyper = self.getModelValue()
        self.optimizer_hyper = self.getOptimValue()
        self.loss_hyper = self.getLossValue()

    def runOneRandomSearch(self):
        self.update_hyper()
        model = self.model_type(self.model_const,self.model_hyper)
        model.compile(self.fit_const,self.optimizer_hyper,self.loss_hyper)
        epoch_dfs = model.fitter.runRandomSearch()
        self.models.append((model,self.model_hyper,self.optimizer_hyper,self.loss_hyper,epoch_dfs))
        return epoch_dfs

    def prepareRandomSearch(self):
        self.update_hyper()
        model = self.model_type(self.model_const,self.model_hyper)
        model.compile(self.fit_const,self.optimizer_hyper,self.loss_hyper)
        epoch_dfs = model.fitter.runPre()
        self.models.append((model, self.model_hyper, self.optimizer_hyper, self.loss_hyper, epoch_dfs))
        return epoch_dfs

    def runKRandomSearchs(self, k ,max_epochs):
        pre_epoch_dfs = self.prepareRandomSearch()
        epoch_runtime = pre_epoch_dfs['runtime'].iloc[-1]
        self.fit_const['max_epochs'] = max_epochs
        max_time = int(max_epochs * epoch_runtime * k)
        permit = input('Time spent for this random search will be less than'+time.strftime('%H:%M:%S',time.gmtime(max_time))+'\ncontinue?[y/n]')
        if permit=='y' or permit=='Y':
            start_time = time.time()
            for i in range(k):
                print('Running {0}th RandomSearch'.format(i))
                self.runOneRandomSearch()
            self.models = self.models[1:]
            end_time = time.time()
            period = end_time - start_time
            print('Time spend:'+time.strftime('%H:%M:%S',time.gmtime(period)))
            self.report()
        else:
            print('RandomSearch Not Engaged')

    def report(self):
        K = len(self.models)
        valid_misclasses, test_misclasses, train_misclasses = np.ones(K), np.ones(K), np.ones(K)
        model_hypers, optim_hypers, loss_hypers = np.zeros((K, len(self.model_hypers))), np.zeros(
            (K, len(self.optimizer_hypers))), np.zeros((K, len(self.loss_hypers)))
        for k, res in enumerate(self.models):
            valid_misclass, train_misclass, test_misclass = res[4]['valid_misclass'].values[-1], \
                                                            res[4]['train_misclass'].values[-1], \
                                                            res[4]['test_misclass'].values[
                                                                -1]
            valid_misclasses[k], train_misclasses[k], test_misclasses[k] = valid_misclass, train_misclass, test_misclass
            for i, key in enumerate(res[1]):
                model_hypers[k, i] = res[1][key]
            for i, key in enumerate(res[2]):
                optim_hypers[k, i] = res[2][key]
            for i, key in enumerate(res[3]):
                loss_hypers[k, i] = res[3][key]
        N_para = len(self.loss_hypers)+len(self.optimizer_hypers)+len(self.model_hypers)
        count = 1
        plt.figure()
        for i,key in enumerate(self.model_hypers):
            plt.subplot(N_para,1,count);count+=1
            plt.plot(model_hypers[:,i])
            plt.title(key)
        for i,key in enumerate(self.optimizer_hypers):
            plt.subplot(N_para, 1, count);count += 1
            plt.plot(optim_hypers[:,i])
            plt.title(key)
        for i,key in enumerate(self.loss_hypers):
            plt.subplot(N_para, 1, count);count += 1
            plt.plot(loss_hypers[:,i])
            plt.title(key)

        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(valid_misclasses)
        plt.title('ValidMisclass')

        plt.subplot(3, 1, 2)
        plt.plot(train_misclasses)
        plt.title('TrainMisclass')

        plt.subplot(3, 1, 3)
        plt.plot(test_misclasses)
        plt.title('TestMisclass')

        self.best_model_idx = int(np.argmin(valid_misclasses))
        self.best_model = self.models[self.best_model_idx][0]
        self.best_model_hyper = self.models[self.best_model_idx][1]
        self.best_optim_hyper = self.models[self.best_model_idx][2]
        self.best_loss_hyper = self.models[self.best_model_idx][3]