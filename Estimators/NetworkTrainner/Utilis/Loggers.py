from abc import ABC, abstractmethod
from visdom import Visdom
import logging


#logger


class Logger(ABC):
    @abstractmethod
    def log_epoch(self,epoch_dfs):
        raise NotImplementedError('Need to implement the log_epoch function')

class Printer(Logger):
    """
    print output of model training using logging
    print last row of result in a epoch
    """
    def log_epoch(self,epoch_dfs,log):
        i_epoch = len(epoch_dfs) - 1
        log.info('Epoch {:d}'.format(i_epoch))
        last_row = epoch_dfs.iloc[-1]
        for key,val in last_row.iteritems():
            log.info('{:25s} {:.5f}'.format(key,val))
        log.info('')

class VisdomWriter(Logger):
    """
    print epoch_dfs in visdom
    """
    def __init__(self,env=None):
        if env==None:
            self.viz = Visdom()
        else:
            self.viz = Visdom(env=env)
        self.viz.line([0.], [0.], win='train_loss', opts=dict(title='训练损失'))
        self.viz.line([0.], [0.], win='valid_loss', opts=dict(title='验证损失'))
        self.viz.line([0.], [0.], win='test_loss', opts=dict(title='测试损失'))
        self.viz.line([0.], [0.], win='train_acc', opts=dict(title='训练准确度'))
        self.viz.line([0.], [0.], win='valid_acc', opts=dict(title='验证准确度'))
        self.viz.line([0.], [0.], win='test_acc', opts=dict(title='测试准确度'))
        self.viz.line([0.], [0.], win='train_avg_acc', opts=dict(title='训练Trial准确度'))
        self.viz.line([0.], [0.], win='valid_avg_acc', opts=dict(title='验证Trial准确度'))
        self.viz.line([0.], [0.], win='test_avg_acc', opts=dict(title='测试Trial准确度'))

    def log_epoch(self,epoch_dfs,env):
        i_epoch = len(epoch_dfs) - 1
        train_loss = epoch_dfs['train_loss'].iloc[-1]
        valid_loss = epoch_dfs['valid_loss'].iloc[-1]
        test_loss = epoch_dfs['test_loss'].iloc[-1]
        train_acc = 1 - epoch_dfs['train_misclass'].iloc[-1]
        valid_acc = 1 - epoch_dfs['valid_misclass'].iloc[-1]
        test_acc = 1 - epoch_dfs['test_misclass'].iloc[-1]
        try:
            train_acc_avg = 1 - epoch_dfs['train_avg_misclass'].iloc[-1]
            valid_acc_avg = 1 - epoch_dfs['valid_avg_misclass'].iloc[-1]
            test_acc_avg = 1 - epoch_dfs['test_avg_misclass'].iloc[-1]
            self.viz.line([train_acc_avg], [i_epoch], win='train_avg_acc', update='append')
            self.viz.line([valid_acc_avg], [i_epoch], win='valid_avg_acc', update='append')
            self.viz.line([test_acc_avg], [i_epoch], win='test_avg_acc', update='append')
        except:
            pass

        self.viz.line([train_loss], [i_epoch], win='train_loss', update='append')
        self.viz.line([valid_loss], [i_epoch], win='valid_loss', update='append')
        self.viz.line([test_loss], [i_epoch], win='test_loss', update='append')
        self.viz.line([train_acc], [i_epoch], win='train_acc', update='append')
        self.viz.line([valid_acc], [i_epoch], win='valid_acc', update='append')
        self.viz.line([test_acc], [i_epoch], win='test_acc', update='append')