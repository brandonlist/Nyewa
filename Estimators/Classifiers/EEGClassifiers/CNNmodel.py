from Estimators.Utilis.TorchExt import Expression,square,safe_log,get_conved_dims
from torch.nn import init
import torch
from Estimators.NetworkTrainner.Fitters.NeuralNetworkFitters import VanillaFitter
import numpy as np
from Estimators.Utilis.TorchExt import make_dot

from torch import nn

class ShallowConvNet(nn.Module):
    def __init__(self,model_const,model_hyper):
        super(ShallowConvNet, self).__init__()
        self.model_const = model_const
        self.model_hyper = model_hyper

        #update const parameters
        self.in_chans = model_const['in_chans']
        self.time_steps = model_const['time_steps']
        self.classes = model_const['classes']
        self.fs = model_const['fs']
        self.final_conv_length = model_const['final_conv_length']

        #update hyper parameters
        self.drop_prob = model_hyper['drop_prob']
        self.linear_init_std = model_hyper['linear_init_std']
        self.pace_1_ratio = model_hyper['pace_1_ratio']
        self.n_filters_time = model_hyper['n_filters_time']
        self.n_filters_spat = model_hyper['n_filters_spat']
        self.pool_kernel_ratio = model_hyper['pool_kernel_ratio']
        self.bn = model_hyper['bn']
        self.relu = model_hyper['relu']
        self.conv_init_bias = model_hyper['conv_init_bias']

        self.pace_1 = round(self.time_steps * self.pace_1_ratio)
        self.pool_kernel = round(self.time_steps*self.pool_kernel_ratio)
        self.pool_stride = 1

        self.cls_network = nn.Sequential()

        self.cls_network.add_module('temp_conv',nn.Conv2d(in_channels=1,out_channels=self.n_filters_time,kernel_size=(1,self.pace_1),stride=1,padding=0))
        self.cls_network.add_module('spat_conv',nn.Conv2d(in_channels=self.n_filters_time,out_channels=self.n_filters_spat,kernel_size=(self.in_chans,1),stride=1))
        if self.relu==True:
            self.cls_network.add_module('relu',nn.ReLU(inplace=True))
        if self.bn==True:
            self.cls_network.add_module('batch_norm',nn.BatchNorm2d(num_features=self.n_filters_spat,track_running_stats=False))
        self.cls_network.add_module('square',Expression(square))
        self.cls_network.add_module('avg_pool',nn.AvgPool2d(kernel_size=(1,self.pool_kernel),stride=(1,self.pool_stride)))
        self.cls_network.add_module('safe_log',Expression(safe_log))
        self.cls_network.add_module('drop_out',nn.Dropout(p=self.drop_prob))
        self.dim = get_conved_dims(torch.Tensor(1,1,self.in_chans,self.time_steps),self.cls_network)
        self.use_auto_length = False
        if self.final_conv_length==0:
            self.use_auto_length = True
            self.final_conv_length = int(self.dim[1])
        self.cls_network.add_module('cls_conv',nn.Conv2d(in_channels=self.n_filters_spat,out_channels=self.classes,kernel_size=(1,self.final_conv_length),bias=True))
        self.cls_network.add_module('softmax',nn.Softmax())


        self.kernels = [1,self.pace_1]
        self.strides = [1,1]

    def weigth_init(self,m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, self.conv_init_bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, self.linear_init_std)
            m.bias.data.zero_()

    def compile(self,fit_const,optimizer_hyper,loss_hyper):
        self.apply(self.weigth_init)
        train_set = fit_const['train_set']
        valid_set = fit_const['valid_set']
        test_set = fit_const['test_set']
        fitter = fit_const['fitter']

        max_epochs = fit_const['max_epochs']
        max_increase_epochs = fit_const['max_increase_epochs']
        cuda = fit_const['cuda']
        batch_size = fit_const['batch_size']

        lr = optimizer_hyper['lr']
        model_constraint = loss_hyper['model_constraint']

        if fitter=='VanillaFitter':
            self.fitter = VanillaFitter(
                self,
                train_set,
                valid_set,
                test_set,
                max_epochs=max_epochs,
                max_increase_epochs=max_increase_epochs,
                cuda=cuda,
                batch_size=batch_size,
                lr=lr,
                model_constraint=model_constraint
            )

    def preprocess(self,x):
        x = (x - x.mean())/x.std()
        return x

    def forward(self, x):
        """
            x:input:[batch_size,1,in_chans,time_steps]
            conv1_kernel:(1,pace_1)
            conv1_output:[batch_size,n_filters_time,in_chans,time_steps-(pace_1-1)]
            sconv_kernel:(in_chans,1)
            sconv_output:[batch_size,n_filters_spat,1,time_steps-(pace_1-1)]
            maxpool_kernel:(1,pool_kernel)
            maxpool_stride:(1,pool_stride)
            maxpool_output:[batch_size,n_filters_spat,1,(time_steps-(pace_1-1)-(pool_kernel-1))/pool_stride)]
            linear_input:n_filters_spat*(time_steps-(pace_1-1)-(pool_kernel-1))/pool_stride)
            linear_output:classes
        """
        if x.dim() != 4:
            x = torch.unsqueeze(x,dim=1)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        batch_size = x.shape[0]
        x = self.preprocess(x)
        x = self.cls_network(x)
        x = x.squeeze(dim=2).squeeze(dim=2)
        return x

    def get_grad(self):
        return [x.grad for x in self.fitter.optimizer.param_groups[0]['params']]

    def get_grad_abs_sum(self):
        grad = self.get_grad(self)
        return np.array([x.abs().sum() for x in grad]).sum()

    def get_grad_sum(self):
        grad = self.get_grad(self)
        return np.array([x.sum() for x in grad]).sum()

    def graph(self):
        dummy = torch.ones(2,1,25,176)
        model_const = {
            'in_chan':25,
            'time_steps':176,
            'classes':4,
            'fs':250,
            'final_conv_length':self.model_const['final_conv_length']
        }
        model_hyper = self.model_hyper
        model = ShallowConvNet(model_const=model_const,model_hyper=model_hyper)
        ans = model(dummy)

        vis_graph = make_dot(model(dummy), params=dict(model.named_parameters()))
        vis_graph.view()