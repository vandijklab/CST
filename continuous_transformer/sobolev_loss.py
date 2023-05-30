import torch
import numpy as np
from torch import nn



def to_np(x):
    return x.detach().cpu().numpy()


class sobolev_loss:
    def __init__(self,k=2,p=2,dim=2,
                 bs=16,data_length=None,
                 minimize=True,diff_mode='central',
                 factor=1,get_derivatives=False,
                 use_mse=True,use_mean=False,
                 frobenius=False):
        self.k=k
        self.p=p
        self.dim=dim
        self.bs=bs
        self.data_length = data_length
        #self.loss0 = F.mse_loss
        self.minimize = minimize
        self.factor=factor
        self.use_mse = use_mse
        self.use_mean = use_mean
        self.frobenius = frobenius
        
        

        self.diff_mode = diff_mode
            
    def evaluate__loss(self,y,data,x,x_fd=None,y_0=None,indexes=None,get_derivatives=False):
        
        if y_0 is None:
            if self.frobenius is False:
                loss = torch.norm(y-data,p=self.p,dim=[i for i in range(1,len(data.shape))]).mean()
            else:
                loss = torch.norm(y-data,p='fro',dim=[-2,-1]).mean()
        else:
            if self.frobenius is False:
                loss = torch.norm(y_0-data,p=self.p,dim=[i for i in range(1,len(data.shape))]).mean()
            else:
                loss = torch.norm(y_0-data,p='fro',dim=[-2,-1]).mean()
        
        if get_derivatives is True:
            loss_derivatives = []
           
        
        for q in range(1,self.k+1):
            if q == 1:
                djydxj = torch.Tensor([]).to(y)
                out = y
            else: 
                out = djydxj[...,-1].to(y)
            

            if q == 1:
                graph_creation=True
            else:
                graph_creation=False
            dydxi = torch.autograd.grad(
                    (out), x,grad_outputs = (torch.ones_like(out)),
                     allow_unused=True,create_graph=graph_creation, retain_graph=True
                    )
            djydxj = torch.cat([djydxj,dydxi[0].unsqueeze(-1)],-1)
            
        
            if self.minimize is True:

                if indexes is None:
                    if self.frobenius is False:
                        loss_q = self.factor*torch.norm(djydxj[...,-1],\
                                                        p=self.p,dim=[i for i in\
                                                        range(1,len(djydxj[...,-1].shape))]).mean()
                    else:
                        loss_q = self.factor*torch.norm(djydxj[...,-1],\
                                                        p='fro',dim=[-2,-1]).mean()
                else:
                    if self.frobenius is False:
                        loss_q = self.factor*torch.norm(djydxj[:,indexes,...,-1],\
                                                        p=self.p,dim=[i for i in\
                                                        range(1,len(djydxj[:,indexes,...,-1].shape))])\
                                                        .mean()
                    else:
                        loss_q = self.factor*torch.norm(djydxj[:,indexes,...,-1],\
                                                        p='fro',dim=[-2,-1]).mean()
                loss += loss_q
                if get_derivatives is True:
                    loss_derivatives.append(loss_q)
                    
                    
            elif self.minimize is False and q==1:
                if self.diff_mode == 'central':
                    average_djydxj =  djydxj[:,:-2,-self.k:] - djydxj[:,2:,-self.k:]
                    average_Ddatadxi = (data[:,:-2,:]-data[:,2:,:])/\
                                       (x_fd[:-2]-x_fd[2:]).unsqueeze(0)\
                                       .repeat(x.shape[0],1).unsqueeze(-1)#.view(x.shape[0],x.shape[1]-2,x.shape[-1])
                    
                elif self.diff_mode == 'forward':
                    average_djydxj = 1/2*(djydxj[:,:-1,-self.k:]+djydxj[:,1:,-self.k:])
                    average_Ddatadxi = torch.diff(data,dim=1)/\
                                       torch.diff(x_fd.clone().detach(),dim=0)#.view(x.shape[0],x.shape[1]-1,x.shape[-1])
                if self.use_mean is True: 
                    loss += self.factor*((torch.norm(average_djydxj-average_Ddatadxi,dim=(1,2))**self.p).mean())
                else:
                    loss += self.factor*((torch.norm(average_djydxj-average_Ddatadxi,dim=(1,2))**self.p).sum()) 
                
            elif self.minimize is False and q==2:
                if self.diff_mode == 'central':
                    average_djydxj =  djydxj[:,:-2,-self.k:] - djydxj[:,2:,-self.k:] 
                    average_Ddatadxi = (data[:,:-2,:]+data[:,2:,:]-2*data[:,1:-1,:])/\
                                       (x_fd[:-2]-x_fd[2:]).unsqueeze(0)\
                                       .repeat(x.shape[0],1).unsqueeze(-1)#.view(x.shape[0],x.shape[1]-2,x.shape[-1])**2
                if self.use_mean is True:
                    loss += self.factor*((torch.norm(average_djydxj-average_Ddatadxi,dim=(1,2))**self.p).mean())
                else:
                    loss += self.factor*((torch.norm(average_djydxj-average_Ddatadxi,dim=(1,2))**self.p).sum())
        
        
        if self.k>0:
            del djydxj, dydxi
            

        if get_derivatives is False:
            return loss
        else:
            return loss, loss_derivatives
        
