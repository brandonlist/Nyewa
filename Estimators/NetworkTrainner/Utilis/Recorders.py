from copy import deepcopy

#recoder
class ModelOptimRecorder(object):
    """
    method
    remember_epoch
        record the parameter of the model and optimizer
    reset_to_best_model
        reset parameter of best model
    """
    def __init__(self,metric):
        self.metric = metric
        self.best_epoch = 0
        self.lowest_val = float('inf')
        self.model_state_dict = None
        self.optimizer_state_dict = None

    def remember_epoch(self,epoch_dfs,model,optimizer,log):
        """
        Remember parameter values in case this epoch
        has the best performance so far(is the lowest).

        Parameters
        ----------
        epoch_dfs: `pandas.Dataframe`
            Dataframe containing the column `metric` with which performance
            is evaluated.
        model: `torch.nn.Module`
            as to load parameters to model
        optimizer: `torch.optim.Optimizer`
            as to load parameters to optimizer
        """
        i_epoch = len(epoch_dfs) - 1
        current_val = float(epoch_dfs[self.metric].iloc[-1])
        if current_val <= self.lowest_val:
            self.best_epoch = i_epoch
            self.lowest_val = current_val
            self.model_state_dict = deepcopy(model.state_dict())
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())
        log.info('Best epoch metric {:s} = {:5f} remembered on epoch {:d}'.format(self.metric,current_val,i_epoch))
        log.info('')

    def reset_to_best_model(self, epoch_dfs, model, optimizer):
        """
        Reset parameters to parameters at best epoch and remove rows
        after best epoch from epochs dataframe.

        Modifies parameters of model and optimizer, changes epoch_dfs in-place.

        Parameters
        ----------
        epoch_dfs: `pandas.Dataframe`
        model: `torch.nn.Module`
        optimizer: `torch.optim.Optimizer`

        """
        # Remove epochs past the best one from epochs dataframe
        epoch_dfs.drop(range(self.best_epoch + 1, len(epoch_dfs)), inplace=True)
        model.load_state_dict(self.model_state_dict)
        optimizer.load_state_dict(self.optimizer_state_dict)