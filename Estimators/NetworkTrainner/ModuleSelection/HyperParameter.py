import numpy as np
import random
from collections import OrderedDict
import time
import matplotlib.pyplot as plt
import copy

def get_random_param(params,range=False,log=None):
    """
    return random value of set in params
    :param params:
    for range==False, params can be:
        test_params=OrderedDict({
        'p1':[1,2,3,4],
        'p2':[1,3,4,5],
        'p3':np.arange(5)
    })
    for range==True, params can be:
        test_params_range=OrderedDict({
            'p1':[1,4],
            'p2':[1.2,5.6],
            'p3':[1e-5,1e-2],
            'p4':[True,False]
        })
    :param range: Should be set to True if only the min and max are provided, Otherwise False
    :param log: Whether to use logspace, can only be used when range is True
    :return:
    """
    #TODO:range can be list indicating which param is using range
    epsilon = 1e-10
    if log==None:
        log = [False] * len(params)
    if range==False:
        values = {}
        for i,key in enumerate(params):
            assert np.array(range).any()==False,print('can only use log space when parameter range is True')
            idx = random.randint(0, len(params[key])-1)
            value = params[key][idx]
            values[key] = value
        return values
    elif range==True:
        values = {}
        for i,key in enumerate(params):
            assert len(params[key])==2,print('please enter 2 nums if using range')
            if log[i]==False:
                if type(params[key][0])==bool:
                    value = random.uniform(0,1) < 0.5
                else:
                    assert params[key][0] < params[key][1], print('first value have to be smaller than the latter one')
                    value = random.uniform(float(params[key][0]),float(params[key][1]))
                    if type(params[key][0])==int:
                        value = int(value)
                values[key] = value
            elif log[i]==True:
                low_bound = np.array(params[key][0])
                high_bound = np.array(params[key][1])
                value = random.uniform(np.log10(low_bound+epsilon),np.log10(high_bound+epsilon))
                value = np.power(10,value)
                values[key] = value
        return values


class RandomSearcher(object):
    """
    model should have method model.fit

    """
    def __init__(self,model_hypers,optimizer_hypers,loss_hypers,model_type):
        self.model_hypers = model_hypers
        self.optimizer_hypers = optimizer_hypers
        self.loss_hypers = loss_hypers
        self.params = model_hypers.copy()
        self.params.update(self.optimizer_hypers)
        self.params.update(self.loss_hypers)

        self.model_type = model_type
        self.models = []


    def getModelValue(self,range,log):
        model_hyper = get_random_param(self.model_hypers,range,log)
        return model_hyper

    def getOptimValue(self,range,log):
        optimizer_hyper = get_random_param(self.optimizer_hypers,range,log)
        return optimizer_hyper

    def getLossValue(self,range,log):
        loss_hyper = get_random_param(self.loss_hypers,range,log)
        return loss_hyper

    #TODO:change it to Search you idiot
    def runOneRandomSearch(self,model_const,model_range,model_log,
                           fit_const,optim_range,optim_log,
                           loss_range,loss_log):
        model_hyper = self.getModelValue(model_range,model_log)
        self.model_hyper = model_hyper
        model = self.model_type(model_const,model_hyper)
        #TODO:check if model works using a input

        optimizer_hyper = self.getOptimValue(optim_range,optim_log)
        self.optimizer_hyper = optimizer_hyper
        loss_hyper = self.getLossValue(loss_range,loss_log)
        self.loss_hyper = loss_hyper
        model.compile(fit_const,optimizer_hyper,loss_hyper)
        epoch_dfs = model.fitter.runRandomSearch()
        self.models.append((model,model_hyper,optimizer_hyper,loss_hyper,epoch_dfs))
        return epoch_dfs

    def prepareRandomSearch(self,model_const,model_range,model_log,
                           fit_const,optim_range,optim_log,
                           loss_range,loss_log):
        model_hyper = self.getModelValue(model_range, model_log)
        self.model_hyper = model_hyper
        model = self.model_type(model_const, model_hyper)
        # TODO:check if model works using a input

        optimizer_hyper = self.getOptimValue(optim_range, optim_log)
        self.optimizer_hyper = optimizer_hyper
        loss_hyper = self.getLossValue(loss_range, loss_log)
        self.loss_hyper = loss_hyper
        model.compile(fit_const, optimizer_hyper, loss_hyper)
        epoch_dfs = model.fitter.runPre()
        self.models.append((model, model_hyper, optimizer_hyper, loss_hyper, epoch_dfs))
        return epoch_dfs

    def runKRandomSearchs(self, model_const, model_range,
                            fit_const, optim_range,
                            loss_range, k, model_log=None,loss_log=None, optim_log=None):
        pre_epoch_dfs = self.prepareRandomSearch(model_const,model_range,model_log,
                           fit_const,optim_range,optim_log,
                           loss_range,loss_log)
        epoch_runtime = pre_epoch_dfs['runtime'].iloc[-1]
        max_epochs = 10
        fit_const['max_epochs'] = max_epochs
        max_time = int(max_epochs * epoch_runtime * k)
        permit = input('Time spent for this random search will be less than'+time.strftime('%H:%M:%S',time.gmtime(max_time))+'\ncontinue?[y/n]')
        if permit=='y' or permit=='Y':
            start_time = time.time()
            for i in range(k):
                print('Running {0}th RandomSearch'.format(k))
                self.runOneRandomSearch(model_const, model_range, model_log,
                                         fit_const, optim_range, optim_log,
                                         loss_range, loss_log)
            self.models = self.models[1:]
            end_time = time.time()
            period = end_time - start_time
            print('Time spend:'+time.strftime('%H:%M:%S',time.gmtime(period)))
            self.report()
        else:
            print('RandomSearch Not Engaged')

    def report(self):
        K = len(self.models)
        valid_misclasses,test_misclasses,train_misclasses = np.ones(K),np.ones(K),np.ones(K)
        for k,res in enumerate(self.models):
            valid_misclass,train_misclass,test_misclass = res[4]['valid_misclass'].values[0],res[4]['train_misclass'].values[0],res[4]['test_misclass'].values[0]
            valid_misclasses[k],train_misclasses[k],test_misclasses[k] = valid_misclass,train_misclass,test_misclass
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