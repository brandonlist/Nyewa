import random
import copy
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np


class SignalAndTarget(Dataset):
    def __init__(self,dataset,subject_ids=None,ts_np=True):
        """

        :param dataset:
        :param subject_ids: None by default, meaning all subjects
        :param ts_np: whether to transfer self.X and self.y to numpy array
        """
        if dataset.type=='BCI':
            self.sub_dataset = copy.deepcopy(dataset)
            self.sub_dataset.subjects_data = []
            if subject_ids==None:
                subject_ids = range(dataset.n_subject)
            for i in subject_ids:
                self.sub_dataset.subjects_data.append(dataset.subjects_data[i])
            (self.X,self.y) = next(iter(DataLoader(self.sub_dataset,len(self.sub_dataset))))
            if ts_np:
                self.X = np.array(self.X)
                self.y = np.array(self.y)
        elif dataset.type=='CV':
            #TODO:make a CVdatabase and changed this to CVClassifyDatabase(you can not distribute code for every single cvdatasets)
            self.sub_dataset = copy.deepcopy(dataset)
            (self.X, self.y) = next(iter(DataLoader(self.sub_dataset, len(self.sub_dataset))))
            if ts_np:
                self.X = np.array(self.X)
                self.y = np.array(self.y)

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.X)

    def __getitem__(self,idx):
        return (self.X[idx],self.y[idx])

    def apply_to_X_y(self,fn):
        st = copy.deepcopy(self)
        st.X = fn(self.X)
        st.y = fn(self.y)
        return st

    def split_into_two_sets(self, first_set_fraction=None, n_first_set=None, shuffle=False):
        """
        Split set into two sets either by fraction of first set or by number
        of trials in first set.

        Parameters
        ----------
        dataset: :class:`.SignalAndTarget`
        first_set_fraction: float, optional
            Fraction of trials in first set.
        n_first_set: int, optional
            Number of trials in first set

        Returns
        -------
        first_set, second_set: :class:`.SignalAndTarget`
            The two splitted sets.
        """
        set = copy.deepcopy(self)
        assert (first_set_fraction is None) != (
                n_first_set is None
        ), "Pass either first_set_fraction or n_first_set plz"
        if n_first_set is None:
            n_first_set = int(round(len(set.X) * first_set_fraction))
        assert n_first_set < len(set.X)
        if shuffle:
            mask = random.sample(list(range(len(set))), n_first_set)
            antimask = [x for x in list(range(len(set))) if x not in mask]
            first_set = set.apply_to_X_y(lambda a: a[mask])
            second_set = set.apply_to_X_y(lambda a: a[antimask])
            del set;return first_set, second_set
        first_set = set.apply_to_X_y(lambda a: a[:n_first_set])
        second_set = set.apply_to_X_y(lambda a: a[n_first_set:])
        del set;return first_set, second_set

    def concatenate_data(self, a, b):
        assert type(a)==type(b),'mismatched types of a and b'
        if type(a)==np.ndarray and type(b)==np.ndarray:
            new = np.concatenate((a, b), axis=0)
        elif type(a)==torch.Tensor:
            new = torch.cat([a,b],dim=0)
        return new

    def concatenate_two_sets(self,other_set):
        set = copy.deepcopy(self)
        new_X = self.concatenate_data(self.X,other_set.X)
        new_y = self.concatenate_data(self.y,other_set.y)
        set.X = new_X
        set.y = new_y
        return set

def concatenate_sets(sets):
    """
    Concatenate all sets together.

    Parameters
    ----------
    sets: list of :class:`.SignalAndTarget`

    Returns
    -------
    concatenated_set: :class:`.SignalAndTarget`
    """
    concatenated_set = sets[0]
    for s in sets[1:]:
        concatenated_set = concatenated_set.concatenate_two_sets(s)
    return concatenated_set