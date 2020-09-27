import torch

def get_conved_dims(x,model):
    ans = model(x)
    return (int(ans.shape[-2]),int(ans.shape[-1]))