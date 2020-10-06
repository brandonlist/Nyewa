import torch
import logging
import time

import pandas as pd
import numpy as np

from collections import OrderedDict

from Estimators.NetworkTrainner.Utilis.Loggers import Printer
from Estimators.NetworkTrainner.Utilis.Recorders import ModelOptimRecorder

logging.basicConfig(format='%(asctime)s  - %(levelname)s: %(message)s',
                    level=logging.INFO)

def np_to_var(
    X, requires_grad=False, dtype=None, **tensor_kwargs
):
    """
    Convenience function to transform numpy array to `torch.Tensor`.

    Converts `X` to ndarray using asarray if necessary.

    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor

    Returns
    -------
    var: `torch.Tensor`
    """
    if not hasattr(X, "__len__"):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = torch.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    return X_tensor


#Base fitter for Neural Networks
class NetworkBaseFitter(object):
    """

    """
    def __init__(self,
                 model,
                 train_set,
                 valid_set,
                 test_set,
                 iterator,
                 loss_function,
                 optimizer,
                 model_constraint,
                 monitors,
                 stop_criterion,
                 regloss_model=None,
                 cuda=False,
                 metric='valid_misclass',
                 ):
        self.__dict__.update(locals())
        del self.self
        self.datasets = OrderedDict(
            (("train", train_set), ("valid", valid_set), ("test", test_set))
        )
        if valid_set is None:
            self.datasets.pop("valid")
        if test_set is None:
            self.datasets.pop("test")
        self.loggers = [Printer()]
        self.log = logging.getLogger(__name__)
        if self.cuda:
            assert torch.cuda.is_available() ,'Cuda not available'
            self.model.cuda()
        #reset if fit again
        self.recorder = ModelOptimRecorder(metric=self.metric)
        #setup epoch_dataframe
        self.epoch_dfs = pd.DataFrame()


    def train_batch(self, inputs, targets):
        """
        Train on given inputs and targets.

        Parameters
        ----------
        inputs: `torch.autograd.Variable`
        targets: `torch.autograd.Variable`
        """
        self.model.train()
        if type(inputs) == np.ndarray:
            inputs = np_to_var(inputs)
        if type(targets) == np.ndarray:
            targets = np_to_var(targets)
        targets = targets.long()
        if self.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, targets)
        if self.regloss_model is not None:
            loss = loss + self.regloss_model(self.model)
        loss.backward()
        self.optimizer.step()
        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)

    def run_one_epoch(self, datasets, remember_best):
        """
        Run training and evaluation on given datasets for one epoch.

        Parameters
        ----------
        datasets: OrderedDict
            Dictionary with train, valid and test as str mapping to
            :class:`.SignalAndTarget` objects.
        remember_best: bool
            Whether to remember parameters if this epoch is best epoch.
        """
        batch_generator = self.iterator.get_batches(
            datasets['train'], shuffle=True
        )
        start_train_epoch_time = time.time()
        for inputs, targets in batch_generator:
            if len(inputs) > 0:
                self.train_batch(inputs, targets)
        end_train_epoch_time = time.time()
        self.log.info(
            "Time spent for training updates: {:.2f}s".format(
                end_train_epoch_time - start_train_epoch_time
            )
        )

        self.monitor_epoch(datasets)
        self.log_epoch()
        if remember_best:
            self.recorder.remember_epoch(
                self.epoch_dfs, self.model, self.optimizer,self.log
            )

    def run_until_stop(self,datasets,remember_best):
        """
        Run trainning and evaluation on given datasets until stop criterion fulfilled
        :param datasets: OrderedDict
        :param remember_best: bool
            whether to remember paras at best epoch
        :return:
        """
        self.monitor_epoch(datasets)
        self.log_epoch()
        if remember_best:
            self.recorder.remember_epoch(self.epoch_dfs,self.model,self.optimizer,self.log)
        self.iterator.reset_rng()
        while not self.stop_criterion.should_stop(self.epoch_dfs):
            self.run_one_epoch(datasets, remember_best)

    def log_epoch(self):
        for logger in self.loggers:
            logger.log_epoch(self.epoch_dfs,self.log)

    def eval_on_batch(self,inputs,targets):
        """

        :param inputs: torch.autograd.Variable
        :param targets: torch.autograd.Variable
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            if type(inputs) == np.ndarray:
                inputs = np_to_var(inputs)
            if type(targets) == np.ndarray:
                targets = np_to_var(targets)
            targets = targets.long()
            if self.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            if hasattr(outputs, "cpu"):
                outputs = outputs.cpu().detach().numpy()
            else:
                # assume it is iterable
                outputs = [o.cpu().detach().numpy() for o in outputs]
            loss = loss.cpu().detach().numpy()
        return outputs, loss

    def monitor_epoch(self,datasets):
        """
        1.evaluate one epoch on given dataset
        2.append epoch_dfs
        :param datasets:
        :return:
        """
        result_dicts_per_monitor = OrderedDict()
        for m in self.monitors:
            result_dicts_per_monitor[m] = OrderedDict()
            result_dict = m.monitor_epoch()
            if result_dict is not None:
                result_dicts_per_monitor[m].update(result_dict)

        for setname in datasets:
            dataset = datasets[setname]
            batch_generator = self.iterator.get_batches(dataset,shuffle=False)

            #get n_batchs
            if hasattr(batch_generator, "__len__"):
                n_batches = len(batch_generator)
            else:
                n_batches = sum(1 for i in batch_generator)
                batch_generator = self.iterator.get_batches(
                    dataset, shuffle=False
                )
            #TODO written to speedup?(need testing) can be simplified
            all_preds,all_targets = None, None
            all_losses,all_batch_sizes = [], []
            for inputs,targets in batch_generator:
                preds, loss = self.eval_on_batch(inputs, targets)
                all_losses.append(loss)
                all_batch_sizes.append(len(targets))
                if all_preds is None:
                    assert all_targets is None
                    if len(preds.shape) == 2:
                        # first batch size is largest
                        max_size, n_classes = preds.shape
                        # pre-allocate memory for all predictions and targets
                        all_preds = np.nan * np.ones(
                            (n_batches * max_size, n_classes), dtype=np.float32
                        )
                    else:
                        assert len(preds.shape) == 3
                        # first batch size is largest
                        max_size, n_classes, n_preds_per_input = preds.shape
                        # pre-allocate memory for all predictions and targets
                        all_preds = np.nan * np.ones(
                            (
                                n_batches * max_size,
                                n_classes,
                                n_preds_per_input,
                            )  ,
                            dtype=np.float32,
                        )
                    all_preds[: len(preds)] = preds
                    all_targets = np.nan * np.ones((n_batches * max_size))
                    all_targets[: len(targets)] = targets
                else:
                    start_i = sum(all_batch_sizes[:-1])
                    stop_i = sum(all_batch_sizes)
                    all_preds[start_i:stop_i] = preds
                    all_targets[start_i:stop_i] = targets
            self.check = all_preds
            # check for unequal batches
            unequal_batches = len(set(all_batch_sizes)) > 1
            all_batch_sizes = sum(all_batch_sizes)
            # remove nan rows in case of unequal batch sizes
            if unequal_batches:
                #will have trouble if network output Nan
                assert np.sum(np.isnan(all_preds[: all_batch_sizes - 1])) == 0,print(all_preds)
                assert np.sum(np.isnan(all_preds[all_batch_sizes:])) > 0,print(all_preds)
                # TODO: is there a reason we dont just take
                # all_preds = all_preds[:all_batch_sizes] and
                # all_targets = all_targets[:all_batch_sizes] ?
                range_to_delete = range(all_batch_sizes, len(all_preds))
                all_preds = np.delete(all_preds, range_to_delete, axis=0)
                all_targets = np.delete(all_targets, range_to_delete, axis=0)
            assert (
                    np.sum(np.isnan(all_preds)) == 0
            ), "There are still nans in predictions"
            assert (
                    np.sum(np.isnan(all_targets)) == 0
            ), "There are still nans in targets"
            # add empty dimension
            # monitors expect n_batches x ...
            all_preds = all_preds[np.newaxis, :]
            all_targets = all_targets[np.newaxis, :]
            all_batch_sizes = [all_batch_sizes]
            all_losses = [all_losses]


            for m in self.monitors:
                result_dict = m.monitor_set(
                    setname,
                    all_preds,
                    all_losses,
                    all_batch_sizes,
                    all_targets,
                    dataset,
                )
                if result_dict is not None:
                    result_dicts_per_monitor[m].update(result_dict)
        row_dict = OrderedDict()
        for m in self.monitors:
            row_dict.update(result_dicts_per_monitor[m])

        #没看懂这个Dataframe为什么要这样搞
        self.epoch_dfs = self.epoch_dfs.append(row_dict, ignore_index=True)
        assert set(self.epoch_dfs.columns) == set(row_dict.keys()), (
            "Columns of dataframe: {:s}\n and keys of dict {:s} not same"
        ).format(str(set(self.epoch_dfs.columns)), str(set(row_dict.keys())))
        self.epoch_dfs = self.epoch_dfs[list(row_dict.keys())]

#Base fitter for Instant Estimators
class InstantBaseFitter(object):
    def __init__(self,
                 model,
                 train_set,
                 valid_set,
                 test_set,
                 metric='acc'):
        self.model = model
        self.metric = metric
        self.datasets = OrderedDict(
            (("train", train_set), ("valid", valid_set), ("test", test_set))
        )
        if valid_set is None:
            self.datasets.pop("valid")
        if test_set is None:
            self.datasets.pop("test")