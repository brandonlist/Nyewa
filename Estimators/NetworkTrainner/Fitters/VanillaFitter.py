from Estimators.NetworkTrainner.Utilis.StopCriterions import Or, MaxEpochs, NoDecrease
from torch import optim
from Estimators.NetworkTrainner.Utilis.Iterators import BalancedBatchSizeIterator
from Estimators.NetworkTrainner.Utilis.Monitors import LossMonitor, MisclassMonitor, RuntimeMonitor
from Estimators.NetworkTrainner.Utilis.Constraints import MaxNormDefaultConstraint
import torch.nn.functional as F
from Estimators.NetworkTrainner.Fitters.BaseFitter import BaseFitter
import copy

class VanillaFitter(BaseFitter):
    def __init__(self,
                 model,
                 train_set,
                 valid_set,
                 test_set,
                 max_epochs,
                 max_increase_epochs,
                 cuda,
                 batch_size,
                 lr,
                 model_constraint
                 ):
        """
        Iterator:BalancedBatchSizeIterator
        monitor:LossMonitor(), MisclassMonitor(), RuntimeMonitor()
        :param model:
        :param train_set:
        :param valid_set:
        :param test_set:
        :param max_epochs:
        :param max_increase_epochs:
        :param cuda:
        :param batch_size:
        :param lr:
        :param model_constraint:
        """
        stop_criterion = Or(
            [
                MaxEpochs(max_epochs),
                NoDecrease("valid_misclass", max_increase_epochs),
            ]
        )
        self.model = model
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        iterator = BalancedBatchSizeIterator(batch_size=batch_size)
        monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
        if model_constraint == True:
            model_constraint = MaxNormDefaultConstraint()
        else:
            model_constraint = None
        loss_function = F.nll_loss
        super(VanillaFitter, self).__init__(model=model,
                                            train_set=train_set,
                                            valid_set=valid_set,
                                            test_set=test_set,
                                            iterator=iterator,
                                            loss_function=loss_function,
                                            optimizer=optimizer,
                                            model_constraint=model_constraint,
                                            monitors=monitors,
                                            stop_criterion=stop_criterion,
                                            cuda=cuda)

    def run(self):
        """
        run trainning
        :return:
        """
        self.log.info('Start Trainning...')
        self.run_until_stop(self.datasets, remember_best=True)
        self.recorder.reset_to_best_model(self.epoch_dfs, self.model, self.optimizer)
        return self.epoch_dfs

    def runPre(self):
        self.log.info('Preparing for RandomSearch')
        self.monitor_epoch(self.datasets)
        self.log_epoch()
        self.run_one_epoch(self.datasets,remember_best=False)
        return self.epoch_dfs

    def runRandomSearch(self):
        res = self.run()
        return res



