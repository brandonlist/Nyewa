import torch
import numpy as np

def GetTorchVersion():
    import torch
    print(torch.__version__)

def GetTorchCudaVersion():
    import torch
    print(torch.version.cuda)

def GetCudnnVersion():
    import torch
    print(torch.backends.cudnn.version())

def GetCudaDeviceName():
    import torch
    print(torch.cuda.get_device_name())



def init_learning():
    """
    1.set random seed so the result of the learning process is replicable
    2.set device
    :return:
    """
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""

def SetCudaDevice():
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def ClearGPUCache():
    import torch
    torch.cuda.empty_cache()

"""

