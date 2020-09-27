import torch
import os
import pickle
import sys
import tkinter as tk
import numpy as np
from torch.utils.data import Dataset
from PIL import Image,ImageTk


class ImageInspector(tk.Frame):
    def __init__(self,datasets):
        super(ImageInspector, self).__init__()
        self.datasets = datasets
        self.master.title('数据集：'+datasets.name)

        self.var = tk.Variable()
        self.var.set(0)
        scale = tk.Scale(self.master, from_=0, to=len(self.datasets), command=self.onScale)
        scale.pack(side=tk.BOTTOM, padx=15)
        self.panel = tk.Label(self.master)
        self.panel.pack(padx=10, pady=10)
        self.label = tk.Entry(self.master)
        self.label.pack(padx=10,pady=10)

    def onScale(self,val):
        self.var.set(int(float(val)))
        img = self.datasets.arrayToImage(self.datasets.datas['imgs'][int(self.var.get())],show=True)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)
        self.label.delete(0, 'end')
        self.label.insert(0, str(self.datasets.classes_dict[self.datasets.datas['labels'][int(self.var.get())]]))

class CIFAR10(Dataset):
    def __init__(self,file_dir=r'G:\CIFAR-10'):
        super(CIFAR10, self).__init__()
        self.type = 'CV'
        self.file_dir = file_dir
        with open(os.path.join(file_dir,'batches.meta'),'rb') as f:
            batches_meta = pickle.load(f,encoding='bytes')
            self.classes_dict = {i: label.decode('utf-8') for i, label in enumerate(batches_meta[b'label_names'])}
        self.train_data_files = ['data_batch_' + str(i+1) for i in range(5)]
        self.test_data_files = ['test_batch']
        self.name = 'CIFAR-10'
        self.datas = {}
        self.datas['imgs'] = []
        self.datas['labels'] = []
        for data_file in self.train_data_files+self.test_data_files:
            data_file = os.path.join(self.file_dir,data_file)
            with open(data_file,'rb') as f:
                batch_data = pickle.load(f,encoding='bytes')
                self.datas['imgs'].append(batch_data[b'data'])
                self.datas['labels'].append(np.array(batch_data[b'labels']))
        self.datas['imgs'] = np.concatenate(self.datas['imgs'])
        self.datas['labels'] = np.concatenate(self.datas['labels'])
        print('CIFAR-10：数据加载完成')

    def inspect_imgs(self):
        root = tk.Tk()
        inspector = ImageInspector(self)
        inspector.mainloop()

    def arrayToImage(self,array,show=False):
        img = array.reshape(3, 32, 32)
        if show:
            img = img.swapaxes(0, 2)
            img = img.swapaxes(0, 1)
        return img

    def memory(self):
        print(sys.getsizeof(self) / 1024 / 1024, 'MB')

    def __getitem__(self, idx):
        img = self.arrayToImage(self.datas['imgs'][idx])
        return (img,self.datas['labels'][idx])

    def __len__(self):
        return len(self.datas['imgs'])


"""
别在这个文件下执行！只是为了演示
from Datasets.CV.CVDatasets import CIFAR10
db = CIFAR10()
from Datasets.Utils.SignalTarget import SignalAndTarget
st = SignalAndTarget(db)
p_st = st.apply_to_X_y(lambda a:a[:50])
t,e = st.split_into_two_sets(0.7,shuffle=True)
from Datasets.Utils.SignalTarget import concatenate_sets
ans = concatenate_sets([t,e])

"""