
""" Utils Functions """

import os
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset

from collections import OrderedDict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_np(x):
    return x.detach().cpu().numpy()

def set_seeds(seed=31):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device

def split_last(x, shape):
    "split the last dimension to given shape"
#     print('x.shape: ',x.shape)
    shape = list(shape)
#     print('shape: ',shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
#     print('*shape: ',*shape)
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def find_sublist(haystack, needle):
    """Return the index at which the sequence needle appears in the
    sequence haystack, or -1 if it is not found, using the Boyer-
    Moore-Horspool algorithm. The elements of needle and haystack must
    be hashable.
    https://codereview.stackexchange.com/questions/19627/finding-sub-list
    """
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if haystack[i - j] != needle[-j - 1]:
                i += skip.get(haystack[i], n)
                break
        else:
            return i - n + 1
    return -1


def get_logger(name, log_path):
    "get logger"
    logger = logging.getLogger(name)
    fomatter = logging.Formatter(
        '[ %(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    if not os.path.isfile(log_path):
        f = open(log_path, "w+")

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)

    logger.setLevel(logging.DEBUG)
    return logger


def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)



class EarlyStopping():

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class SaveBestModel:

    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

        
    # def __call__(self, current_valid_loss, epoch, model, kernel, ode_func = None):
    def __call__(self, path, current_valid_loss, epoch, model, G_NN = None, kernel=None, F_func = None, f_func=None):
        if current_valid_loss < self.best_valid_loss:
            
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            
            if G_NN is not None: G_NN_state = {'state_dict': G_NN.state_dict()}
            if kernel is not None: kernel_state = {'state_dict': kernel.state_dict()}
            if F_func is not None: F_func_state = {'state_dict': F_func.state_dict()}
            if f_func is not None: f_func_state = {'state_dict': f_func.state_dict()}
            
            torch.save(model, os.path.join(path,'model.pt'))
            if G_NN is not None:
                torch.save(G_NN_state, os.path.join(path,'G_NN.pt'))
                if f_func is not None: torch.save(f_func_state, os.path.join(path,'f_func.pt'))
            else:
                if kernel is not None: torch.save(kernel_state, os.path.join(path,'kernel.pt'))
                if F_func is not None: torch.save(F_func_state, os.path.join(path,'F_func.pt'))
                if f_func is not None: torch.save(f_func_state, os.path.join(path,'f_func.pt'))

def load_checkpoint(path, G_NN, optimizer, scheduler, kernel, F_func, f_func=None):
    print('Loading ', os.path.join(path))
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
     
    checkpoint = torch.load(os.path.join(path, 'model.pt'), map_location=map_location)
    if G_NN is not None: 
        start_epoch = checkpoint['epoch']
        offset = start_epoch
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    if G_NN is not None: 
        checkpoint = torch.load(os.path.join(path, 'G_NN.pt'), map_location=map_location)
        G_NN.load_state_dict(checkpoint['state_dict'])
    
    if kernel is not None: 
        checkpoint = torch.load(os.path.join(path, 'kernel.pt'), map_location=map_location)
        kernel.load_state_dict(checkpoint['state_dict'])
    if F_func is not None: 
        checkpoint = torch.load(os.path.join(path, 'F_func.pt'), map_location=map_location)
        F_func.load_state_dict(checkpoint['state_dict'])
    if f_func is not None: 
        checkpoint = torch.load(os.path.join(path, 'f_func.pt'), map_location=map_location)
        f_func.load_state_dict(checkpoint['state_dict'])
    
    if G_NN is not None: 
        return G_NN, optimizer, scheduler, kernel, F_func, f_func
    else: 
        return checkpoint


        
class EarlyStopping_2pBERT:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model1,  string_name, path_to_save):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model1, string_name, path_to_save)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model1, string_name, path_to_save)
            self.counter = 0

    def save_checkpoint(self, val_loss, model1, string_name, path_to_save):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model1.state_dict(), os.path.join(path_to_save,'checkpoint_' + string_name + '_BERT.pt'))
        self.val_loss_min = val_loss


class fun_interpolation():
    def __init__(self,y,points, verbose=False, given_points=None, start_point=None):
        self.y = y
        self.points = points
        self.verbose = verbose
        self.given_points = given_points
        self.start_point=start_point
        
        if self.verbose: 
            print('self.points: ',self.points)
            print('self.y.shape: ',self.y.shape)
    
    def step_interpolation(self,x):
        values = self.y[:,0,:][:,None,:] # Assign the first point
        if self.verbose: 
            print('values.shape: ',values.shape)

        for i in range(x.size(0)-1):
            all_dist = torch.abs(x[i] - self.points)
            min_idx = all_dist.argmin()
            if self.verbose:
                print('all_dist: ',all_dist)
                print('min_idx: ',all_dist.argmin())
                print('values.shape: ',values.shape)
                print('self.y[:,min_idx,:][:,None,:].shape: ',self.y[:,min_idx,:][:,None,:].shape)
            values = torch.cat((values,self.y[:,min_idx,:][:,None,:]),dim=1)

        return values
    
    def linear_interpolation(self,x):
        x = x.squeeze()
        batch_values = torch.zeros(self.y.size(0),x.size(0),self.y.size(2)) #[batch_size, number_points, dim]
        if self.verbose: 
            print('batch_values.shape: ',batch_values.shape)
            
        t_lin = self.points.repeat(self.y.size(2),1)
        if self.verbose: print('t_lin.shape: ',t_lin.shape)

        for idx_batch in range(batch_values.size(0)):
            x_lin = self.y[idx_batch,:].squeeze().T
            # y = y_orig[:,:-2]
            if self.verbose: print('x_lin.shape: ',x_lin.shape)

            t_in_lin = x.repeat(self.y.size(2),1)
            if self.verbose: print('t_in_lin.shape: ',t_in_lin.shape)

            yq_cpu = Interp1d()(t_lin, x_lin, t_in_lin, None)
            if self.verbose: 
                print('yq_cpu.T.shape: ',yq_cpu.T.shape)
            batch_values[idx_batch,:] = yq_cpu.T            
        
        return batch_values
    
    def spline_interpolation(self,x):
        coeffs = natural_cubic_spline_coeffs(self.points, self.y)
        spline = NaturalCubicSpline(coeffs)
        out = spline.evaluate(x)
        return out

    def cte_2nd_half(self, x, noise=None, c_scaling_factor=1):
        # frame_to_drop = int(self.y.shape[1]/2)
        frame_to_drop = self.given_points
        out = torch.zeros_like(self.y)
        # print('out.shape: ',out.shape)
        # print('torch.cat((self.y[:,:frame_to_drop,:], self.y[:,frame_to_drop-1,:].repeat(self.y[:,frame_to_drop:,:].shape[0],1)),dim=0).shape: ',torch.cat((self.y[:,:frame_to_drop,:], self.y[:,frame_to_drop-1,:].repeat(self.y[:,frame_to_drop:,:].shape[0],1)),dim=0).shape)
        for idx_batch in range(out.shape[0]):
            if self.verbose: 
                print('self.y[idx_batch,frame_to_drop:,:].shape: ',self.y[idx_batch,frame_to_drop:,:].shape)
                print('self.y[idx_batch,:frame_to_drop,:].shape: ',self.y[idx_batch,:frame_to_drop,:].shape)
                print('self.y[idx_batch,frame_to_drop,:]: ',self.y[idx_batch,frame_to_drop,:])
                
            if noise is not None:
                out[idx_batch,:] = (torch.cat((self.y[idx_batch,:frame_to_drop,:], self.y[idx_batch,frame_to_drop-1,:].repeat(self.y[idx_batch,frame_to_drop:,:].shape[0],1)),dim=0)+noise[idx_batch,:])/c_scaling_factor
            else: 
                out[idx_batch,:] = torch.cat((self.y[idx_batch,:frame_to_drop,:], self.y[idx_batch,frame_to_drop-1,:].repeat(self.y[idx_batch,frame_to_drop:,:].shape[0],1)),dim=0)/c_scaling_factor
        
        return out

    def cte_2nd_half_shifted(self, x, std=0):
        frame_to_drop = self.given_points
        out = torch.zeros_like(self.y)
        for idx_batch in range(out.shape[0]):
            if self.verbose: 
                print('self.y[idx_batch,frame_to_drop:,:].shape: ',self.y[idx_batch,frame_to_drop:,:].shape)
                print('self.y[idx_batch,:frame_to_drop,:].shape: ',self.y[idx_batch,:frame_to_drop,:].shape)
                print('self.y[idx_batch,frame_to_drop,:]: ',self.y[idx_batch,frame_to_drop,:])
                
            if std>0:
                print('adding perturbation: ')
                perturb = torch.normal(mean=torch.zeros_like(self.y[idx_batch,frame_to_drop:,:]),
                                          std=std)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                out[idx_batch,:] = torch.cat((self.y[idx_batch,:frame_to_drop,:], self.y[idx_batch,frame_to_drop-1,:].repeat(self.y[idx_batch,frame_to_drop:,:].shape[0],1)+perturb),dim=0)
            else: 
                if self.start_point is None:
                    out[idx_batch,:] = torch.cat((self.y[idx_batch,:frame_to_drop,:], self.y[idx_batch,frame_to_drop-1,:].repeat(self.y[idx_batch,frame_to_drop:,:].shape[0],1)),dim=0)
                else: #In this case, shift the given points by 'start_point'
                    out[idx_batch,:] = torch.cat((self.y[idx_batch,self.start_point,:].repeat(self.y[idx_batch,:self.start_point,:].shape[0],1), 
                                                  self.y[idx_batch,self.start_point:self.start_point+frame_to_drop,:], 
                                                  self.y[idx_batch,self.start_point+frame_to_drop-1,:].repeat(self.y[idx_batch,self.start_point+frame_to_drop:,:].shape[0],1)),dim=0)
        
        return out

class Train_val_split:
    def __init__(self, IDs,val_size_fraction):
        
        
        IDs = np.random.permutation(IDs)
        # print('IDs: ',IDs)
        self.IDs = IDs
        self.val_size = int(val_size_fraction*len(IDs))
    
    def train_IDs(self):
        train = sorted(self.IDs[:len(self.IDs)-self.val_size])
        return train
    
    def val_IDs(self):
        val = sorted(self.IDs[len(self.IDs)-self.val_size:])
        return val

class Train_val_split3:
    '''
    In this class, each frame is a new curve
    '''
    def __init__(self, IDs,val_size_fraction, segment_len,segment_window_factor):
        verbose=True
        #Split the data into len(data)/(1.5*batch_size)

        # bins = np.arange(0,len(IDs),step=np.ceil(bins),dtype=np.dtype(np.int16))[:-1] #Discard the last ID because it might lead to smaller sequence
        bins = np.arange(len(IDs)-segment_len,dtype=np.dtype(np.int16))[:-1]
        if verbose: print("bins: ",bins)

        IDs = np.random.permutation(bins) 
        if verbose: print('IDs:', IDs)

        val_size = int(val_size_fraction*len(IDs))
        
        self.IDs = IDs
        self.val_size = int(np.ceil(val_size_fraction*len(IDs)))
    
    def train_IDs(self):
        train = sorted(self.IDs[:len(self.IDs)-self.val_size])
        return train
    
    def val_IDs(self):
        val = sorted(self.IDs[len(self.IDs)-self.val_size:])
        return val

class Dynamics_Dataset3(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, Data, times, frames_to_drop):
        'Initialization'
        self.times = times.float()
        self.Data = Data.float()
        self.frames_to_drop = frames_to_drop

    def __getitem__(self, index):
        ID = index 
        obs = self.Data[ID]
        t = self.times 
        frames_to_drop = self.frames_to_drop[index]

        return obs, t, ID, frames_to_drop 
    
    def __len__(self):
        'Denotes the total number of points'
        return len(self.times)

class Dynamics_Dataset4(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, Data, Data_orig, times, times_orig, frames_to_drop):
        'Initialization'
        self.times = times.float()
        self.times_orig = times_orig.float()
        self.Data = Data.float()
        self.Data_orig = Data_orig.float()
        self.frames_to_drop = frames_to_drop

    def __getitem__(self, index):

        ID = index #self.list_IDs[index]
        obs = self.Data[ID]
        obs_orig = self.Data_orig[ID]
        t = self.times #Because it already set the number of points in the main script
        t_orig = self.times_orig #Because it already set the number of points in the main script
        frames_to_drop = self.frames_to_drop[index]

        return obs, obs_orig, t, t_orig, ID, frames_to_drop 
    
    def __len__(self):
        'Denotes the total number of points'
        return len(self.times)

class Dynamics_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, Data, times):
        'Initialization'
        self.times = times.float()
        self.Data = Data.float()
        # self.batch_size = batch_size

    def __getitem__(self, index):
        ID = index #self.list_IDs[index]
        obs = self.Data[ID]
        t = self.times

        return obs, t, ID, ID #Just adding one more output so I can drop later 
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.times)

def plot_dim_vs_time(obs_to_print, time_to_print, z_to_print, dummy_times_to_print, z_all_to_print, frames_to_drop, path_to_save_plots, name, epoch, args):
    verbose=False
    if verbose: 
        print('[plot_dim_vs_time] obs_to_print.shape: ',obs_to_print.shape)
        print('[plot_dim_vs_time] time_to_print.shape: ',time_to_print.shape)
        print('[plot_dim_vs_time] args.num_dim_plot: ',args.num_dim_plot)
        print('[plot_dim_vs_time] dummy_times_to_print.shape: ',dummy_times_to_print.shape)
        print('[plot_dim_vs_time] z_all_to_print.shape: ',z_all_to_print.shape)
        
        
    n_plots_x = int(np.ceil(np.sqrt(args.num_dim_plot)))
    n_plots_y = int(np.floor(np.sqrt(args.num_dim_plot)))
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=(10, 10), sharex=True, dpi=100, facecolor='w', edgecolor='k')
    ax=ax.ravel()
    for idx in range(args.num_dim_plot):
        ax[idx].plot(dummy_times_to_print,z_all_to_print[:,idx],c='r', label='model')
        if frames_to_drop is not None and frames_to_drop>0:
            ax[idx].scatter(time_to_print[:-frames_to_drop],obs_to_print[:-frames_to_drop,idx],label='Data',c='blue', alpha=0.5)
            ax[idx].scatter(time_to_print[-frames_to_drop:],obs_to_print[-frames_to_drop:,idx],label='Hidden',c='green', alpha=0.5)
        else:
            ax[idx].scatter(time_to_print[:],obs_to_print[:,idx],label='Data',c='blue', alpha=0.5)
        ax[idx].set_xlabel("Time")
        ax[idx].set_ylabel("dim"+str(idx))
        ax[idx].legend()
    fig.tight_layout()

    if args.mode=='train' or path_to_save_plots is not None:
        plt.savefig(os.path.join(path_to_save_plots, name + str(epoch)))
        plt.close('all')
    else: plt.show()
    
    del obs_to_print, time_to_print, z_to_print, frames_to_drop
    
    
class SaveBestModel_CST:

    def __init__(
        self, best_valid_loss=float('inf'), verbose=True
    ):
        self.best_valid_loss = best_valid_loss
        self.verbose = verbose
        
    def __call__(self, path, current_valid_loss, epoch, model):
        if current_valid_loss < self.best_valid_loss:
            
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch}\n")
            
            torch.save(model, os.path.join(path,'model.ckpt'))
            
        else:
            if self.verbose is True:
                print(f"\nLower validation loss still: {self.best_valid_loss}") 
            