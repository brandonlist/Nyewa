from torch import nn
from Estimators.Utilis.TorchExt import get_conved_dims
import torch
from torch.nn import init
from Estimators.NetworkTrainner.Fitters.NeuralNetworkFitters import VanillaFitter

class LeNet5(nn.Module):
    """

    """
    def __init__(self,model_const,model_hyper):
        """

        :param model_const:parameter of model that won't hurt model's performance, have sth to do with dataset
        :param model_hyper: parameter of the model that will influence model's performance
        """
        super(LeNet5,self).__init__()
        self.img_size = model_const['img_size']
        self.in_chan = model_const['in_chan']
        self.n_classes = model_const['n_classes']

        self.n_kernel_0 = model_hyper['n_kernel_0']
        self.n_kernel_1 = model_hyper['n_kernel_1']

        self.linear_init_std = model_hyper['linear_init_std']
        self.conv_init_bias = model_hyper['conv_init_bias']

        self.Conv_1=nn.Sequential(nn.Conv2d(self.in_chan,self.n_kernel_0,kernel_size=5,stride=1,padding=0),
                                 nn.AvgPool2d(kernel_size=2,stride=2),
                                 nn.Conv2d(self.n_kernel_0,self.n_kernel_1,kernel_size=5,stride=1,padding=0),
                                 nn.AvgPool2d(kernel_size=2,stride=2),
                                 )
        self.h,self.w = get_conved_dims(torch.ones(1,self.in_chan,self.img_size[0],self.img_size[1]),self.Conv_1)


        self.Fc_1=nn.Sequential(nn.Linear(self.n_kernel_1*self.h*self.w,120),
                                nn.ReLU(),
                                nn.Linear(120,84),
                                nn.ReLU(),
                                nn.Linear(84,self.n_classes),
                                nn.Softmax())
    # self.criteria=nn.CrossEntropyLoss()
    def forward(self, x):
        """

        :param x:[b,in_channel,32,32]
        :return:
        """
        x = x.to(dtype=torch.float32)
        x = self.Conv_1(x)
        x = x.view(-1,self.w*self.h*self.n_kernel_1)
        logits = self.Fc_1(x)
        return logits
        #loss = self.criteria(logits,y)

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





