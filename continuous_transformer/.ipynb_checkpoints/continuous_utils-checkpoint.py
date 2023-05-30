import os, random, torch, h5py
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SequentialSampler, SubsetRandomSampler
from skimage.transform import rescale, resize
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from scipy import interpolate
# from scipy import linalg as la
# import math as m

# import umap
import pickle

from continuous_transformer.utils import Train_val_split, Dynamics_Dataset3, Dynamics_Dataset4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_nvidia_smi():
    bashCommand = "nvidia-smi"
    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output.decode('ascii'))
    
def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def vn_eig_entropy(rho):
    
    rho = rho + 1e-10
    S= torch.mean(-torch.sum(rho*torch.log(rho),dim=1))
        
    return S


def patch_sampling(data, T_orig, patch_sampling_cfg, frame_size, patch_size, current_epoch=None, T_fixed=None):
    sampling_type = patch_sampling_cfg["structure"]
    
    T, P_row, P_col = sampling_full_frames(data, T_orig, patch_sampling_cfg, frame_size, patch_size, T_fixed)
    
    return sampling_segments(data, T, T_orig, P_row, P_col, patch_sampling_cfg, frame_size, patch_size)



def sampling_full_frames(data, T, patch_sampling_cfg, frame_size, patch_size, T_fixed):
    # -- select T samples with replacement for patches
    verbose=False
    num_in_between_frames = patch_sampling_cfg["num_in_between_frames"]
    if num_in_between_frames>0:
        sampling_type = patch_sampling_cfg["sampling_type"]
        
    if T is None: 
        T = torch.arange(0,data.shape[1]-1) # Select T samples with 
    
    if verbose: print('[sampling_full_frames] T: ',T)
    
    #Add in-between frames
    if num_in_between_frames > 0:
        if sampling_type == 'random':
            if T_fixed is None:
                in_between_frames,_ = torch.sort(torch.FloatTensor(num_in_between_frames).uniform_(T[0], T[-1]))
                T_tmp,_ = torch.sort(torch.cat((T,in_between_frames))) # append the in-between and sort
            else: # In this case, T_fixed already has some coordinates that we will include, so just sample the difference
                in_between_frames,_ = torch.sort(torch.FloatTensor(num_in_between_frames-T_fixed.size(0)).uniform_(T[0], T[-1]))
                T_tmp,_ = torch.sort(torch.cat((T,T_fixed,in_between_frames))) # append the in-between and sort


        else: 
            # just get the original coordinates of the data
            T_tmp = torch.linspace(T[0], T[-1], steps=160) # just get the original coordinates of the data
            ids_downsampled = np.linspace(0,len(T_tmp)-1,num=num_in_between_frames+len(T), dtype=np.int64)
            T_tmp = T_tmp[ids_downsampled]
            idx_overlapping = [np.argmin(np.abs(T_tmp.numpy()-T[idx].numpy())) for idx in range(len(T))] #Select the indexes that match the first 5 points used before
            idx_non_overlapping = np.argwhere(np.isin(np.arange(len(T_tmp)),idx_overlapping,invert=True)).flatten()
            in_between_frames = T_tmp[idx_non_overlapping]
            T_tmp,_ = torch.sort(torch.cat((T,in_between_frames))) # append the in-between and sort
            
        
        #Check if there are duplicates and resample if there are
        dup=np.array([0])
        while dup.size != 0:
            u, c = np.unique(T_tmp, return_counts=True)
            dup = u[c > 1]
            if dup.size != 0:
                print('[sampling_full_frames] There are duplicated time coordinates: ',dup)
                print('Resampling')
                
                #Add in-between frames
                if T_fixed is None:
                    in_between_frames,_ = torch.sort(torch.FloatTensor(num_in_between_frames).uniform_(T[0], T[-1]))
                    T_tmp,_ = torch.sort(torch.cat((T,in_between_frames))) # append the in-between and sort
                else:
                    in_between_frames,_ = torch.sort(torch.FloatTensor(num_in_between_frames-T_fixed.size(0)).uniform_(T[0], T[-1]))
                    T_tmp,_ = torch.sort(torch.cat((T,T_fixed,in_between_frames))) # append the in-between and sort
                
        
        T = T_tmp
        del T_tmp
    
    P_row = torch.zeros_like(T,dtype=torch.int64)
    P_col = torch.zeros_like(T,dtype=torch.int64)
    
    return T, P_row, P_col
    
def sampling_random(data, patch_sampling_cfg, frame_size, patch_size):
    # -- sample random patches
    num_patches = patch_sampling_cfg["num_patches"]
    n_pred = patch_sampling_cfg["num_patches_to_hide"]
#     num_frames = patch_sampling_cfg["num_frames"]

    # -- select T coordinates (one for each patch)
    if data.shape[1]> num_patches:
        rep_param = False
    else:
        rep_param = True
        
    T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[1]-1), num_patches-n_pred, replace=rep_param))) # Select T samples with 
    
    P_row = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
    P_col = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
    
    T_lastFrame = torch.ones(n_pred,dtype=torch.int64)*(data.shape[1]-1)
    P_row_lastFrame = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=n_pred)))
    P_col_lastFrame = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=n_pred)))
    
    T = torch.cat((T,T_lastFrame))
    P_row = torch.cat((P_row,P_row_lastFrame))
    P_col = torch.cat((P_col,P_col_lastFrame))

    return T, P_row, P_col

def sampling_grid(data, patch_sampling_cfg, frame_size, patch_size,current_epoch=None):
# def sampling_grid(data, hparams, frame_size, patch_size):
    # -- sample patches in grid-like form
    num_patches = patch_sampling_cfg["num_patches"]
    num_frames = patch_sampling_cfg["num_frames"]
    mode = patch_sampling_cfg["mode"]
#     grid_rows = frame_size//patch_size

    if data.shape[1]> num_patches:
        rep_param = False
    else:
        rep_param = True

    if mode=="inference":
        T = torch.arange(0,data.shape[1]-1) # Select T samples with 
        T = torch.cat((T,torch.tensor([data.shape[1]-1]))) # Append the last frame manually 
        P_row = torch.linspace(0,frame_size["rows"]-patch_size-1,steps=int(frame_size["rows"]/patch_size),dtype=torch.int64)
        P_col = torch.linspace(0,frame_size["cols"]-patch_size-1,steps=int(frame_size["cols"]/patch_size),dtype=torch.int64)
        T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Needs to invert x and y to match actual 
        T,P_row,P_col = T.flatten(),P_row.flatten(),P_col.flatten()
            
        
    else: 
        # To sample a grid from the frame and make sure that last frame is always sampled, because we want to predict that one everytime
        T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[1]-1), num_patches-1, replace=rep_param))) # Select T samples with 
        T = torch.cat((T,torch.tensor([data.shape[1]-1]))) # Append the last frame manually 
        P_row = torch.linspace(0,frame_size["rows"]-patch_size-1,steps=int(frame_size["rows"]/patch_size),dtype=torch.int64)
        P_col = torch.linspace(0,frame_size["cols"]-patch_size-1,steps=int(frame_size["cols"]/patch_size),dtype=torch.int64)
        T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Needs to invert x and y to match actual 
        T,P_row,P_col = T.flatten(),P_row.flatten(),P_col.flatten()
    
    return T, P_row, P_col

def sampling_segments(data, T, T_orig, P_row, P_col, patch_sampling_cfg, frame_size, patch_size):
    verbose=False

    num_patches = patch_sampling_cfg["num_patches"]
    num_frames = patch_sampling_cfg["num_frames"] # NUmber of frames in teh segment
    n_pred = patch_sampling_cfg["num_patches_to_hide"] #Number of patches to hide, 
    n_frames_to_hide = patch_sampling_cfg["n_frames_to_hide"] #Number of frames to hide, 
    in_between_frame_init = patch_sampling_cfg["in_between_frame_init"]
    num_in_between_frames = patch_sampling_cfg["num_in_between_frames"]
    prob_replace_masked_token = patch_sampling_cfg["prob_replace_masked_token"] #prob of replacing a masked token. Previously it was 0.8.
    masking_type = patch_sampling_cfg["masking_type"]
    mode = patch_sampling_cfg["mode"]
    batch_size_segments = data.shape[0]
    if in_between_frame_init == 'interpolation':
        interpolation_kind = patch_sampling_cfg["interpolation_kind"]

    segm_frames = torch.zeros((batch_size_segments,T.shape[0],patch_size[0]*patch_size[1]),dtype=torch.float32) # Using patches. Original for square patches
    
    dict_special_tokens = {
        "MASK": 0.5
    }
    MASK_FRAME = torch.tensor(np.ones(patch_size[0]*patch_size[1])*dict_special_tokens["MASK"], requires_grad=True)
    # in_between_frame_init= 'interpolation'
    # interpolation_kind='linear'
    if in_between_frame_init == 'interpolation':
        if verbose: print('Using interpolation for the initialization')
        data_flat = torch.flatten(data, start_dim=2)
        tmp_time = np.sort(np.random.choice(np.arange(T_orig.shape[0]), n_frames_to_hide, replace=False)) # Select 'n_pred' frames that will be used to compute the interpolating function
        

        # Using scioy for the interpolation
        t_lin= T_orig[tmp_time]#.repeat(data_flat.size(2),1).to(device)
        if verbose: print('t_lin: ',t_lin)
        x_lin = data_flat[:,tmp_time]
        if verbose: print('x_lin: ',x_lin)
        f = interpolate.interp1d(t_lin, x_lin.cpu().detach().numpy(),axis=1,kind=str(interpolation_kind)) #Time is on the axis=1
        
        segm_frames = torch.Tensor(f(T)).to(device)
        if verbose: print('segm_frames.shape: ',segm_frames.shape)  
        del t_lin, x_lin, tmp_time

        
    elif in_between_frame_init != 'interpolation':
        segm_frames = MASK_FRAME*torch.ones((batch_size_segments,T.shape[0],patch_size[0]*patch_size[1]),dtype=torch.float32) # Initializ with mask
        # print('in_between_frame_init: ',in_between_frame_init)
        for idx1 in range(batch_size_segments):
            for idx in range(len(T)):
                # if verbose: print('patch_size: ',patch_size)
                # Only replace the coordinate in segm_frames if the T coordinate is in T_orig. Otherwise, already replace it with MASK_FRAME
                if T[idx] in T_orig:
                    if verbose: print('using the real coordinates: {}, T[idx]: {}'.format(idx,T[idx]))
                    tmp_idx = (T_orig == T[idx]).nonzero(as_tuple=True)[0]
                    segm_frames[idx1,idx, :] = data[idx1,tmp_idx, P_row[tmp_idx]:P_row[tmp_idx]+patch_size[0], P_col[tmp_idx]:P_col[tmp_idx]+patch_size[1]].flatten()
                else:
                    if in_between_frame_init == 'mask':
                        if verbose: print('Masking idx with cte: {}, T[idx]: {}'.format(idx,T[idx]))
                        segm_frames[idx1, idx, :] = MASK_FRAME
                    elif in_between_frame_init == 'random':
                        if verbose: print('Masking idx with random: {}, T[idx]: {}'.format(idx,T[idx]))
                        segm_frames[idx1, idx, :] = torch.FloatTensor(patch_size[0], patch_size[1]).uniform_(-1.0, 1.0).squeeze()

                # if verbose: print('Not masking idx: {}, T[idx]: {}'.format(idx,T[idx]))
     
    
    if in_between_frame_init == 'interpolation':
        n_pred = data.shape[1]
        cand_pos = np.argwhere(np.isin(T.numpy(),T_orig)).flatten() 
        tmp_masked_tokens = torch.empty(batch_size_segments, n_pred, segm_frames.shape[2]) 
    

    else: 

        if n_pred==1 and n_frames_to_hide==1: # mask just the entire whole frame (ie, no patches)
            cand_pos = np.arange(segm_frames.shape[1]-n_pred, segm_frames.shape[1]) #This only masks the last "n_pred" frames
            tmp_masked_tokens = torch.empty(batch_size_segments, n_pred, segm_frames.shape[2]) 
        elif n_pred==1 and n_frames_to_hide>1: # Mask the randomly "n_frames_to_hide" frames, but the whole frame
            if masking_type=='last_n_points': #For last n frames  
                cand_pos = np.arange(segm_frames.shape[1]-n_frames_to_hide, segm_frames.shape[1]) #This only masks the last "n_pred" frames
            elif masking_type=='equal_spaced': # For equally spaced seen points
                n_frames_to_keep = segm_frames.shape[1]-n_frames_to_hide
                cand_pos = np.argwhere(np.isin(np.arange(len(T)),np.floor(np.linspace(0,len(T),n_frames_to_keep)),invert=False)).flatten()
            elif masking_type=='random_masking':
                if mode=='inference': #manually remove 

                    #For the tests using random time points
                    idx_real_coordinates = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                    random.shuffle(idx_real_coordinates)
                    cand_pos = idx_real_coordinates[:3]

                else: 


                    if num_in_between_frames>0: #In this case, we need to make sure only the real coordinates are selected for masking
                        idx_real_coordinates = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                        random.shuffle(idx_real_coordinates)
                        cand_pos = idx_real_coordinates[:n_frames_to_hide]

                    else: #in this case there are no dummy points
                        cand_pos = np.arange(0,len(T)) #Original
                        # cand_pos = np.tile(np.linspace(0,len(T_orig)-1,num=10, dtype=np.int64),(len(T_orig)-1,1))[0] #For the sobolev on the 2D spirals
                        random.shuffle(cand_pos)
                        if verbose: print('cand_pos: ',cand_pos)


            tmp_masked_tokens = torch.empty(batch_size_segments, n_frames_to_hide, segm_frames.shape[2]) 
            n_pred=n_frames_to_hide
            # cand_pos = np.sort(cand_pos) #original
            cand_pos = np.sort(cand_pos[:n_pred]) #original

    tmp_masked_pos = []

    if in_between_frame_init == 'interpolation': #If interpolation is being used, we want to predict all the frames. So copy them to 'masked_tokens'
        for idx1 in range(batch_size_segments):
            for pos, idx_pos in zip(cand_pos[:n_pred], range(n_pred)):
                tmp_masked_tokens[idx1, idx_pos, :] = data[idx1, idx_pos, :].flatten() #Get the tokens from the input data (ie, before inteporlation), since this is what we are trying to predict   
                tmp_masked_pos.append(pos)
    else: 
        if verbose: print('Using masking. The is in_between_frame_init = ', in_between_frame_init)
        
        # Function to mask some of the real frames for prediction
        for idx1 in range(batch_size_segments):
            for pos, idx_pos in zip(cand_pos[:n_pred], range(n_pred)):
                if verbose: print('tmp_masked_tokens[{}, {}, :] = segm_frames[{}, {}, :]'.format(idx1, idx_pos,idx1,pos))
                tmp_masked_tokens[idx1, idx_pos, :] = segm_frames[idx1, pos, :]    
                tmp_masked_pos.append(pos)
                prob = random.random()
                if prob < prob_replace_masked_token: # 80% randomly change token to mask token
                    segm_frames[idx1, pos, :] = MASK_FRAME
                elif prob < 0.9: # 10% randomly change token to random token of the data
                    rand_curve = np.random.randint(batch_size_segments)
                    if verbose: print('rand_curve: ',rand_curve)
                    rand_token = np.random.randint(data.shape[1]) #Before
                    if verbose: print('rand_token: ',rand_token)
                    segm_frames[idx1, pos, :] = data[rand_curve, rand_token, :].squeeze(-1)
                # 10% randomly change token to current token, which in this case is the default

                    
    masked_pos = torch.tensor(np.array(tmp_masked_pos, dtype=np.int64)).reshape(batch_size_segments,n_pred)
    masked_pos = torch.from_numpy(np.array(masked_pos))
    
    masked_tokens = tmp_masked_tokens.reshape(batch_size_segments,n_pred, patch_size[0], patch_size[1])
    if verbose: print('[sampling_segments] masked_tokens.shape: ', masked_tokens.shape)
    masked_tokens = masked_tokens.type_as(data)

    return T, P_row, P_col, segm_frames, masked_pos, masked_tokens


def plot_training_curves(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path):
    # block displaying plots during training
    plt.ioff()

    tmp_size = np.minimum(masked_tokens.shape[0], 10)
    fig, ax = plt.subplots(tmp_size, 2, figsize=(10,50))

    ax = ax.ravel()
    c=0
    tmp_maskpos = masked_pos.detach().cpu().numpy().flatten()
    
    fig, (ax,ax2) = plt.subplots(1,2,figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
    ax.plot(val_loss.detach().cpu())
    ax.set_yscale('log')
    ax.legend(['Validation'], fontsize=15)
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15) 
    ax.set_title('Total losses (Contrast labels)', fontsize=15)
    ax.set_aspect('auto')

    ax2.plot(val_r2)
    ax2.legend(['Val_LM'], fontsize=15)
    ax2.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('R^2', fontsize=15) 
    ax2.set_title('R^2', fontsize=15)
    ax2.set_aspect('auto')
    plt.show()
        
    plt.savefig(os.path.join(save_path, "plots", f"training-curves-epoch-{epoch:05d}.png"))
    plt.close('all')

def plot_attention_weights(scores,epoch, save_path, mode): 
    from matplotlib.gridspec import GridSpec

    verbose=False
    scores = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
    all_att = scores[:,0,:]#.cpu().detach().numpy() # Select just the first curve
    if verbose: print('all_att.shape: ',all_att.shape)
    del scores
    
    mean_att_per_patch_over_time = all_att.mean(axis=(0))
    
    # create objects
    fig = plt.figure(figsize=(10, 5), facecolor='w')
    gs = GridSpec(nrows=2, ncols=4)

    idx_head=0
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(mean_att_per_patch_over_time[idx_head,:].cpu().detach().numpy())
    ax0.set_title('head ' + str(idx_head)+',ent: ' + str(vn_eig_entropy(mean_att_per_patch_over_time[idx_head,:]).cpu().detach().numpy()))

    idx_head=1
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(mean_att_per_patch_over_time[idx_head,:].cpu().detach().numpy())
    ax1.set_title('head ' + str(idx_head)+',ent: ' + str(vn_eig_entropy(mean_att_per_patch_over_time[idx_head,:]).cpu().detach().numpy()))

    idx_head=2
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(mean_att_per_patch_over_time[idx_head,:].cpu().detach().numpy())
    ax2.set_title('head ' + str(idx_head)+',ent: ' + str(vn_eig_entropy(mean_att_per_patch_over_time[idx_head,:]).cpu().detach().numpy()))

    idx_head=3
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(mean_att_per_patch_over_time[idx_head,:].cpu().detach().numpy())
    ax3.set_title('head ' + str(idx_head) +',ent: ' + str(vn_eig_entropy(mean_att_per_patch_over_time[idx_head,:]).cpu().detach().numpy()))

    ax4 = fig.add_subplot(gs[:, 2:])
    mean_att_per_patch_over_time = all_att.mean(axis=(0,1))
    ax4.imshow(mean_att_per_patch_over_time.cpu().detach().numpy())
    ax4.set_title('Mean head&layers, ent: ' + str(vn_eig_entropy(mean_att_per_patch_over_time).cpu().detach().numpy()) )
    
    fig.tight_layout()

    if save_path is not None: 
        if mode=='inference':
            plt.savefig(os.path.join(save_path, "plots", f"att-weight-epoch-{epoch:05d}-{mode}.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"att-weight-epoch-{epoch:05d}-{mode}.png"))
        plt.close('all')
    else: 
        plt.show()
    
def plot_predictions_curves(data,T, T_orig,P_row, P_col, masked_tokens, masked_pos, logits_lm, dev_logits_lm, logits_lm_with_dummy, patch_size,
                     epoch, val_loss, val_r2, save_path, mode):
    # block displaying plots during training
    verbose=False
    data = data.squeeze(-1)
    if masked_tokens is not None: masked_tokens = masked_tokens.squeeze(-1)
     
    plt.ioff()
    if verbose:
        print('[plot_predictions_curves] masked_tokens.shape: ',masked_tokens.shape)

    patches_to_plot = 32 
        
    if masked_tokens is None:
        ref_data = data
    else:
        ref_data = masked_tokens
    tmp_size = np.minimum(ref_data.shape[-1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
    if tmp_size > 2:
        n_cols = int(np.ceil(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
        n_rows = int(np.floor(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
        if n_cols*n_rows<ref_data.shape[1]: n_rows+=1
        fig, ax = plt.subplots(n_rows,n_cols, figsize=(10,5), facecolor='w')
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w')

    ax = ax.ravel()
    c=0
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    
    if ref_data.shape[-1] > patches_to_plot:
        list_idx = np.sort(np.random.choice(np.arange(ref_data.shape[-1]), tmp_size, replace=True))
    else: 
        list_idx = np.arange(ref_data.shape[-1])

    for idx_tmp in list_idx:
        ax[c].scatter(T_orig, data[0,:,idx_tmp].detach().cpu(), label='Data')
        if masked_tokens is not None:
            ax[c].scatter(T[masked_pos[0,:]], masked_tokens[0,:,idx_tmp].detach().cpu(), label='Hidden Data') # For regular Transformer training
        ax[c].set_title("Dim " + str(idx_tmp))

        if logits_lm_with_dummy is not None and len(T)==len(logits_lm_with_dummy[0,:]):
            if mode=='inference':
                ax[c].scatter(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model',alpha=0.2)
                ax[c].plot(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='r', label='model')
            else: 
                ax[c].scatter(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model',alpha=0.2)
        else: 
            ax[c].plot(T[masked_pos[0,:]], logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model')
        if dev_logits_lm is not None:
            ax[c].plot(T, dev_logits_lm[0,:,idx_tmp].squeeze(), label='dy/dt')
        fig.tight_layout()
        ax[c].legend()
        c+=1


    if save_path is not None: 
        if mode=='inference':
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        elif val_loss is None:
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}-loss-{val_loss:.6f}-{mode}-r2-{val_r2:.2f}.png"))
        plt.close('all')
    else: 
        plt.show()
    del dev_logits_lm

def plot_predictions_curves2(data,T, T_orig,data_all,T_all,P_row, P_col, masked_tokens, masked_pos, logits_lm, dev_logits_lm, logits_lm_with_dummy, patch_size,
                     epoch, val_loss, val_r2, save_path, in_between_frame_init, mode):
    # block displaying plots during training
    verbose=False
    data = data.squeeze(-1)
    if masked_tokens is not None: masked_tokens = masked_tokens.squeeze(-1)
    if logits_lm_with_dummy is not None:
        logits_lm_with_dummy = logits_lm_with_dummy.squeeze(-1)
     
    plt.ioff()
    if verbose:
        print('[plot_predictions_curves] masked_tokens.shape: ',masked_tokens.shape)

    patches_to_plot = 32 
        
    if masked_tokens is None:
        ref_data = data
    else:
        ref_data = masked_tokens
    tmp_size = np.minimum(ref_data.shape[-1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
    if tmp_size > 2:
        n_cols = int(np.ceil(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
        n_rows = int(np.floor(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
        if n_cols*n_rows<ref_data.shape[1]: n_rows+=1
        fig, ax = plt.subplots(n_rows,n_cols, figsize=(10,5), facecolor='w')
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w')

    ax = ax.ravel()
    c=0
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    
    if ref_data.shape[-1] > patches_to_plot:
        list_idx = np.sort(np.random.choice(np.arange(ref_data.shape[-1]), tmp_size, replace=True))
    else: 
        list_idx = np.arange(ref_data.shape[-1])

    for idx_tmp in list_idx:
        ax[c].scatter(T_all, data_all[0,:,idx_tmp].detach().cpu(),c='g', label='Data',alpha=0.2) 
        if masked_tokens is not None:
            if in_between_frame_init=='mask':
                ax[c].scatter(T_orig, data[0,:,idx_tmp].detach().cpu(), label='Sampled') # For the BERT framework
                ax[c].scatter(T[masked_pos[0,:]], masked_tokens[0,:,idx_tmp].detach().cpu(), label='Hidden Data') #For the BERT framework
            elif in_between_frame_init=='interpolation':
                ax[c].scatter(T_orig, masked_tokens[0,:,idx_tmp].detach().cpu(), label='Sampled') # For the CST framework
        ax[c].set_title("Dim " + str(idx_tmp))

        if logits_lm_with_dummy is not None and len(T)==len(logits_lm_with_dummy[0,:]):
            if mode=='inference':
                ax[c].plot(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='r', label='model')
            else: 
                ax[c].scatter(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model',alpha=0.2)
        else: 
            ax[c].plot(T[masked_pos[0,:]], logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model')
        if dev_logits_lm is not None:
            ax[c].plot(T, dev_logits_lm[0,:,idx_tmp].squeeze(), label='dy/dt')
        fig.tight_layout()
        ax[c].legend()
        c+=1


    if save_path is not None: 
        if mode=='inference':
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        elif val_loss is None:
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}-loss-{val_loss:.6f}-{mode}-r2-{val_r2:.2f}.png"))
        plt.close('all')
    else: 
        plt.show()
    del dev_logits_lm



def plot_predictions(data,T,T_orig, P_row, P_col, masked_tokens, masked_pos,logits_lm, 
    logits_lm_with_dummy, patch_size,epoch, val_loss, val_r2, range_imshow, save_path, mode):
    verbose=False
    # block displaying plots during training
    plt.ioff()
    
    #verbose=True
    if verbose:
        print('[plot_predictions] T.shape: ',T.shape)
        print('[plot_predictions] T: ',T)
        print('[plot_predictions] T_orig: ',T_orig)
        print('[plot_predictions] data.shape: ',data.shape)
        print('[plot_predictions] P_row.shape: ',P_row.shape)
        print('[plot_predictions] masked_tokens.shape: ',masked_tokens.shape)
        print('[plot_predictions] masked_pos.shape: ',masked_pos.shape)
        if logits_lm_with_dummy is not None:
            print('[plot_predictions_curves] logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)
        

    patches_to_plot = 36 

    if logits_lm_with_dummy is None:
        ref_data = data
    else:
        ref_data = logits_lm_with_dummy
        
    if data.shape[1]==logits_lm_with_dummy.shape[1]: #In this case, we are plotting all the points of the input sequence. No dummies
        fig, ax = plt.subplots(2,data.shape[1], figsize=(30,5), facecolor='w')
        for idx_tmp in range(data.shape[1]):
            ax[0,idx_tmp].imshow(data[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
            if idx_tmp in masked_pos:
                ax[0,idx_tmp].set_title(f"data[{idx_tmp}], t: {T[idx_tmp]:.2f}",color= 'r')#, row: {P_row[idx_tmp]}, col: {P_col[idx_tmp]}")
            else: 
                ax[0,idx_tmp].set_title(f"data[{idx_tmp}], t: {T[idx_tmp]:.2f}")#, row: {P_row[idx_tmp]}, col: {P_col[idx_tmp]}")
            ax[0,idx_tmp].set_xticks([])
            ax[0,idx_tmp].set_yticks([])

            A = logits_lm[0,idx_tmp,:].detach().cpu().numpy().flatten()
            B = data[0,idx_tmp,:].detach().cpu().numpy().flatten()
            r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2


            ax[1,idx_tmp].imshow(logits_lm[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
            ax[1,idx_tmp].set_title(f"logits_lm[{idx_tmp}], r2: {r2_sq:.2f}")
            ax[1,idx_tmp].set_xticks([])
            ax[1,idx_tmp].set_yticks([])

    else: 
        tmp_size = np.minimum(2*ref_data.shape[1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
        if verbose: print('[plot_predictions] tmp_size: ',tmp_size)
        if tmp_size > 2:
            n_cols = int(np.ceil(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
            n_rows = int(np.ceil(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
            if verbose: print('[plot_predictions] n_cols: {}, n_rows: {}'.format(n_cols,n_rows))
            if n_cols*n_rows<2*ref_data.shape[1]: 
                n_rows+=1
            if verbose: print('[plot_predictions] n_cols: {}, n_rows: {}'.format(n_cols,n_rows))
            fig, ax = plt.subplots(n_rows,n_cols, figsize=(50,50), facecolor='w')
        else:
            fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w')

        ax = ax.ravel()
        c=0
        tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
        
        if ref_data.shape[1] > patches_to_plot:
            list_idx = np.sort(np.random.choice(np.arange(masked_tokens.shape[1]), tmp_size, replace=True))
        else: 
            list_idx = np.arange(ref_data.shape[1])
        if verbose: print('list_idx: ',list_idx)
        
        for idx_tmp in list_idx:
            if logits_lm_with_dummy is not None and len(T)==len(logits_lm_with_dummy[0,:]):
                ax[c].imshow(logits_lm_with_dummy[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
                ax[c].set_title("logit[{}], t: {:.3f}, row: {}, col: {}".format(idx_tmp,T[idx_tmp],P_row[idx_tmp],P_col[idx_tmp]),fontsize=18)
                ax[c].set_xticks([])
                ax[c].set_yticks([])
                c+=1
    if save_path is not None: 
        if mode=='inference':
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        elif val_loss is None:
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}-loss-{val_loss:.6f}-{mode}-r2-{val_r2:.2f}.png"))
    plt.close('all')
    
def plot_predictions_in_between(data,T, P_row, P_col, masked_pos, masked_pos_real, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path, replace_by_real_frame=False):
    # block displaying plots during training
    plt.ioff()
    
    logits_lm_copy = logits_lm.detach().cpu().numpy()#.copy()

    patches_to_plot = 36 
            
    if logits_lm_copy.shape[-1]==1 : 
        logits_lm_copy = np.transpose(logits_lm_copy, (0, 2, 1, 3))
        
    tmp_size = np.minimum(logits_lm_copy.shape[1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
    if tmp_size > 2:
        n_rows = int(np.ceil(np.sqrt(tmp_size)))
        n_cols = int(np.floor(np.sqrt(tmp_size)))
        fig, ax = plt.subplots(n_rows,n_cols, figsize=(50,50), facecolor='w')
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w')
        n_rows,n_cols=2,1
    tmp_size = np.minimum(tmp_size, n_rows*n_cols) 

    ax = ax.ravel()
    c=0
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    
    dummy_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=True)).flatten() #returns the indexes of the dummy frames
    real_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=False)).flatten() #returns the indexes of the dummy frames
    pred_idx = masked_pos_real[0]
    
    real_idxs2 = real_idxs[np.argwhere(np.isin(real_idxs,pred_idx,invert=True)).flatten()]
    if real_idxs2.size == 0: real_idxs2 = pred_idx # In this case, all the frames are being predicted
    if replace_by_real_frame:
        fig.suptitle('With True Frames', fontsize=16)
        for idx_tmp in real_idxs2:
            if logits_lm_copy.shape[-1]==1 : 
                logits_lm_copy[0,:,idx_tmp] = data[0,int(T[idx_tmp]),:].detach().cpu().numpy()
            else:     
                logits_lm_copy[0,idx_tmp] = data[0,int(T[idx_tmp]),:].detach().cpu().numpy()
    else: 
        fig.suptitle('Reconst. Frames', fontsize=16)

    if logits_lm_copy.shape[-1]==1 : 
        for idx_tmp in range(tmp_size): #masked_pos.shape[1]):
            ax[c].scatter(T, logits_lm_copy[0,idx_tmp,:])
            ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
            ax[c].set_xticks([])
            ax[c].set_yticks([])
            c+=1
           
    else:
        for idx_tmp in range(tmp_size): #masked_pos.shape[1]):
            ax[c].imshow(logits_lm_copy[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]), vmin=range_imshow[0], vmax=range_imshow[1])
            ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
            ax[c].set_xticks([])
            ax[c].set_yticks([])
            c+=1
    
    if save_path is not None: 
        if replace_by_real_frame:
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-epoch-{epoch:05d}_TrueFrames.png"))
        else:
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-epoch-{epoch:05d}_ReconsFrames.png"))
        plt.close('all')
    
def plot_predictions_in_between_pca(data, T, P_row, P_col, masked_pos, masked_pos_real, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path, replace_by_real_frame=False):
    # block displaying plots during training
    plt.ioff()
    pca = PCA(n_components=2)
    
    logits_lm_copy = logits_lm.detach().cpu().numpy()#.copy()

    patches_to_plot = 36 
    
    fig, ax = plt.subplots(2,2, figsize=(10,10),facecolor='w')

    ax = ax.ravel()
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    
    dummy_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=True)).flatten() #returns the indexes of the dummy frames
    real_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=False)).flatten() #returns the indexes of the real frames
    pred_idx = masked_pos_real[0]
    
    real_idxs2 = real_idxs[np.argwhere(np.isin(real_idxs,pred_idx,invert=True)).flatten()] #returns the indexes of the real frames that were not used for prediction
    if real_idxs2.size == 0: real_idxs2 = pred_idx # In this case, all the frames are being predicted
    
    if replace_by_real_frame:
        fig.suptitle('With True Frames', fontsize=16)
        for idx_tmp in real_idxs2:
            logits_lm_copy[0,idx_tmp] = data[0,int(T[idx_tmp]),:].detach().cpu().numpy()
    else:
        fig.suptitle('Reconst. Frames', fontsize=16)
        
    logits_lm_pca = pca.fit_transform(logits_lm_copy[0].reshape(logits_lm[0].shape[0],-1))  
    ax[0].scatter(logits_lm_pca[dummy_idxs,0], logits_lm_pca[dummy_idxs,1], label='Dummy', c='blue', alpha=0.5)
    ax[0].scatter(logits_lm_pca[real_idxs2,0], logits_lm_pca[real_idxs2,1], label='Data', c='green')
    ax[0].scatter(logits_lm_pca[pred_idx,0], logits_lm_pca[pred_idx,1], label='Pred', c='red')
    ax[0].set_title('All points')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].legend()

    ax[2].scatter(logits_lm_pca[:,0], logits_lm_pca[:,1], c=T, alpha=0.5)
    ax[2].set_title('All points')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    all_idxs = np.concatenate((dummy_idxs,masked_pos_real[0]))
    ax[1].scatter(logits_lm_pca[dummy_idxs,0], logits_lm_pca[dummy_idxs,1], label='Dummy', c='blue', alpha=0.5)
    ax[1].scatter(logits_lm_pca[masked_pos_real[0],0], logits_lm_pca[masked_pos_real[0],1], label='Data', c='red')
    ax[1].set_title('Dummy and Pred')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].legend()

    ax[3].scatter(logits_lm_pca[all_idxs,0], logits_lm_pca[all_idxs,1], c=T[all_idxs], alpha=0.5)
    ax[3].set_title('Dummy and Pred')
    ax[3].set_xticks([])
    ax[3].set_yticks([])
        
    if save_path is not None: 
        if replace_by_real_frame:
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca_epoch-{epoch:05d}_TrueFrames.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca_epoch-{epoch:05d}_ReconsFrames.png"))
        plt.close('all')
    
    del pca, logits_lm_pca, all_idxs
    
def plot_predictions_in_between_pca_vs_time(data, T, P_row, P_col, masked_pos, masked_pos_real, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path, num_components, replace_by_real_frame=False):
    # block displaying plots during training
    plt.ioff()
    if num_components is not None:
        pca = PCA(n_components=num_components)
    
    logits_lm_copy = logits_lm.detach().cpu().numpy()#.copy()

    patches_to_plot = 36 

    tmp_size = np.minimum(num_components, patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
    if tmp_size > 2:
        n_rows = int(np.ceil(np.sqrt(tmp_size)))
        n_cols = int(np.floor(np.sqrt(tmp_size)))
        fig, ax = plt.subplots(n_rows,n_cols, figsize=(20,20), facecolor='w', sharex=True)
    else:
        fig, ax = plt.subplots(6,6, figsize=(20,20),facecolor='w',sharex=True)

    ax = ax.ravel()
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    
    dummy_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=True)).flatten() #returns the indexes of the dummy frames
    real_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=False)).flatten() #returns the indexes of the real frames
    pred_idx = masked_pos_real[0]
    
    real_idxs2 = real_idxs[np.argwhere(np.isin(real_idxs,pred_idx,invert=True)).flatten()] #returns the indexes of the real frames that were not used for prediction
    if real_idxs2.size == 0: real_idxs2 = pred_idx # In this case, all the frames are being predicted
    
    if replace_by_real_frame:
        fig.suptitle('With True Frames', fontsize=16)
        for idx_tmp in real_idxs2:
            logits_lm_copy[0,idx_tmp] = data[0,int(T[idx_tmp]),:].detach().cpu().numpy()
    else:
        fig.suptitle('Reconst. Frames', fontsize=16)
    
    if num_components is not None:
        logits_lm_pca = pca.fit_transform(logits_lm_copy[0].reshape(logits_lm[0].shape[0],-1))

    for idx in range(tmp_size):
    #     
        ax[idx].scatter(T[dummy_idxs], logits_lm_pca[dummy_idxs,idx], label='Dummy', c='blue', alpha=0.5)
        ax[idx].scatter(T[real_idxs2], logits_lm_pca[real_idxs2,idx], label='Data', c='green')
        ax[idx].scatter(T[pred_idx], logits_lm_pca[pred_idx,idx], label='Pred', c='red')
        ax[idx].set_title('PC' + str(idx))
        ax[idx].set_xlabel('T')
        
    if save_path is not None: 
        if replace_by_real_frame:
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca_vs_time_epoch-{epoch:05d}_TrueFrames.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca_vs_time_epoch-{epoch:05d}_ReconsFrames.png"))
        plt.close('all')
    fig.tight_layout()
    
    del pca, logits_lm_pca
    
def plot_predictions_in_between_umap(data, T, P_row, P_col, masked_pos, masked_pos_real, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path, replace_by_real_frame=False):
    # block displaying plots during training
    plt.ioff()
    reducer = umap.UMAP(n_components=2)
    
    logits_lm_copy = logits_lm.detach().cpu().numpy()#.copy()

    patches_to_plot = 36 

    fig, ax = plt.subplots(2,2, figsize=(10,10),facecolor='w')

    ax = ax.ravel()
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    
    dummy_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=True)).flatten() #returns the indexes of the dummy frames
    real_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=False)).flatten() #returns the indexes of the dummy frames
    pred_idx = masked_pos_real[0]
    
    real_idxs2 = real_idxs[np.argwhere(np.isin(real_idxs,pred_idx,invert=True)).flatten()]
    if real_idxs2.size == 0: real_idxs2 = pred_idx # In this case, all the frames are being predicted
    
    if replace_by_real_frame:
        for idx_tmp in real_idxs2:
            logits_lm_copy[0,idx_tmp] = data[0,int(T[idx_tmp]),:].detach().cpu().numpy()
        
    logits_lm_umap = reducer.fit_transform(logits_lm_copy[0].reshape(logits_lm_copy[0].shape[0],-1))
   
    ax[0].scatter(logits_lm_umap[dummy_idxs,0], logits_lm_umap[dummy_idxs,1], label='Dummy', c='blue', alpha=0.5)
    ax[0].scatter(logits_lm_umap[real_idxs2,0], logits_lm_umap[real_idxs2,1], label='Data', c='green')
    ax[0].scatter(logits_lm_umap[pred_idx,0], logits_lm_umap[pred_idx,1], label='Pred', c='red')
    ax[0].set_title('All points')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].legend()

    ax[2].scatter(logits_lm_umap[:,0], logits_lm_umap[:,1], c=T, alpha=0.5)
    ax[2].set_title('All points')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    all_idxs = np.concatenate((dummy_idxs,masked_pos_real[0]))

    ax[1].scatter(logits_lm_umap[dummy_idxs,0], logits_lm_umap[dummy_idxs,1], label='Dummy', c='blue', alpha=0.5)
    ax[1].scatter(logits_lm_umap[masked_pos_real[0],0], logits_lm_umap[masked_pos_real[0],1], label='Data', c='red')
    ax[1].set_title('Dummy and Pred')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].legend()

    ax[3].scatter(logits_lm_umap[all_idxs,0], logits_lm_umap[all_idxs,1], c=T[all_idxs], alpha=0.5)
    ax[3].set_title('Dummy and Pred')
    ax[3].set_xticks([])
    ax[3].set_yticks([])
        
    if save_path is not None: 
        if replace_by_real_frame:
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-umap_epoch-{epoch:05d}_TrueFrames.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-umap_epoch-{epoch:05d}_ReconsFrame.png"))
        plt.close('all')
        
def plot_whole_sequence(data,epoch, range_imshow, save_path): ## (data, self.current_epoch, self.logger.log_dir)
    # block displaying plots during training
    plt.ioff()

    patches_to_plot = 32 
    
    if data.shape[-1]==1: #In this case, juts plot as a time series
        tmp_size = np.minimum(data.shape[2], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
        if tmp_size > 2:
            n_rows = int(np.ceil(np.sqrt(tmp_size)))
            n_cols = int(np.floor(np.sqrt(tmp_size)))
            fig, ax = plt.subplots(n_rows,n_cols, figsize=(50,50), facecolor='w',sharex=True)
        else:
            fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w',sharex=True)
    else: 
        tmp_size = np.minimum(data.shape[1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
        if tmp_size > 2:
            n_rows = int(np.ceil(np.sqrt(tmp_size)))
            n_cols = int(np.floor(np.sqrt(tmp_size)))
            fig, ax = plt.subplots(n_rows,n_cols, figsize=(50,50), facecolor='w',sharex=True)
        else:
            fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w',sharex=True)

    ax = ax.ravel()
    c=0
    tmp_maskpos = data[0,:].detach().cpu().numpy().flatten()

    if data.shape[-1]==1: #In this case, juts plot as a time series
        for idx_tmp in range(data.shape[2]):
            ax[c].scatter(np.arange(len(data[0,:,idx_tmp,:])), data[0,:,idx_tmp,:].detach().cpu())
            ax[c].set_title(f"data[{idx_tmp}]")#, t: {T[tmp_maskpos[idx_tmp]]}")
            ax[c].set_xticks([])
            ax[c].set_yticks([])
            c+=1
        
    else: 
        for idx_tmp in range(data.shape[1]):
            ax[c].imshow(data[0,idx_tmp,:].detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
            ax[c].set_title(f"data[{idx_tmp}]")#, t: {T[tmp_maskpos[idx_tmp]]}")
            ax[c].set_xticks([])
            ax[c].set_yticks([])
            c+=1

    if save_path is not None: 
        plt.savefig(os.path.join(save_path, "plots", f"whole-seq-samples-epoch-{epoch:05d}.png"))
        plt.close('all')
    
def plot_whole_frame(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,frame_size,
                     epoch, val_loss, val_r2, range_imshow, save_path):
    
    fig,ax = plt.subplots(1,2, facecolor='w',figsize=(10,6))
    
    tmp_img1 = torch.zeros(frame_size["rows"],frame_size["cols"])
    count=-int((frame_size["rows"]/patch_size[0])*(frame_size["cols"]/patch_size[1]))
    for idx_patch1 in torch.linspace(0,frame_size["rows"]-patch_size[0],steps=int(frame_size["rows"]/patch_size[0]),dtype=torch.int64):
        for idx_patch2 in torch.linspace(0,frame_size["cols"]-patch_size[1],steps=int(frame_size["cols"]/patch_size[1]), dtype=torch.int64):
            tmp_img1[idx_patch1:idx_patch1+patch_size[0], idx_patch2:idx_patch2+patch_size[1]] = logits_lm[0,count,:].reshape(patch_size[0],patch_size[1])
            count+=1

    ax[1].imshow(tmp_img1,vmin=range_imshow[0], vmax=range_imshow[1])
    count=-int((frame_size["rows"]/patch_size[0])*(frame_size["cols"]/patch_size[1]))

    ax[1].set_title('Reconstructed (Frame: {})'.format(T[-1]))
    
    tmp_img = torch.zeros(frame_size["rows"],frame_size["cols"])
    count= -int((frame_size["rows"]/patch_size[0])*(frame_size["cols"]/patch_size[1]))
    for idx_patch1 in torch.linspace(0,frame_size["rows"]-patch_size[0],steps=int(frame_size["rows"]/patch_size[0]),dtype=torch.int64):
        for idx_patch2 in torch.linspace(0,frame_size["cols"]-patch_size[1],steps=int(frame_size["cols"]/patch_size[1]), dtype=torch.int64):
            tmp_img[idx_patch1:idx_patch1+patch_size[0], idx_patch2:idx_patch2+patch_size[1]] = masked_tokens[0,count,:].reshape(patch_size[0],patch_size[1])
            count+=1

    ax[0].imshow(tmp_img, vmin=range_imshow[0], vmax=range_imshow[1])
    ax[0].set_title('Original')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    r_square_frames = (np.corrcoef(tmp_img1.flatten(), 
                                   tmp_img.flatten(), rowvar=False)[0,1])**2
    fig.suptitle('R2 = {:.4f}'.format(r_square_frames))

    if save_path is not None: plt.savefig(os.path.join(save_path, "plots", f"whole_frame-epoch-{epoch:05d}-val-loss-{val_loss:.6f}-val-r2-{r_square_frames:.3f}.png"))
    
    reconstrcuted_frame = tmp_img1.clone()
    
    return reconstrcuted_frame



def create_dataloaders_toydata(path_file, experiment_name, use_first_n_frames, batch_size_segments, validation_split, regularly_sampled, downsample_points,args, verbose=False):
    
    print('Loading ',os.path.join(path_file,experiment_name + ".p"))
    Data_dict = pickle.load(open(os.path.join(path_file,experiment_name + ".p"), "rb" )) #This data was saved in GPU. So transform it to CPU first
    print(Data_dict.keys())
    Data = Data_dict['Data_orig'][:,:use_first_n_frames,:, None]

    if args.add_noise_to_input_data>0:
        for idx_curve in range(Data.shape[0]):
            noise_0 = np.random.normal(0,args.add_noise_to_input_data,Data.shape[1])
            noise_1 = np.random.normal(0,args.add_noise_to_input_data,Data.shape[1])
            Data[idx_curve,:,0,0] = Data[idx_curve,:,0,0]+noise_0
            Data[idx_curve,:,1,0] = Data[idx_curve,:,1,0]+noise_1

    if downsample_points < Data.shape[1]: #check if there is a need for downsampling
        if regularly_sampled:
            ids_downsampled = np.tile(np.linspace(0,Data.shape[1]-1,num=downsample_points, dtype=np.int64),(Data.shape[0],1))
        else: 
            random_idxs = np.zeros([Data.shape[0],9])
            for idx_tmp in range(Data.shape[0]):
                random_idxs[idx_tmp,:] = np.random.choice(np.arange(1, Data.shape[1],dtype=np.int64), size=downsample_points-1, replace=False)

            ids_downsampled = np.concatenate((np.zeros([Data.shape[0],1],dtype=np.int8),random_idxs),axis=1).astype(int)

            ids_downsampled = np.sort(ids_downsampled)

    scaling_factor = np.quantile(np.abs(Data),0.98)

    train_val = 20000 # Number of frames for train and validation. The remaining will be for test
    n_steps = 3000 #number of iterations for training. default=3k epochs

    Data_test = Data[train_val:,:]

    n_points = Data.shape[1]
    extrapolation_points = Data.shape[1]

    t_max=1
    t_min=0

    index_np = np.arange(0, len(Data), 1, dtype=int)
    index_np = np.hstack(index_np[:, None])
    times_np = np.linspace(t_min, t_max, num=n_points) #Original
    times_np = np.hstack([times_np[:, None]])

    ###########################################################
    times = torch.from_numpy(times_np[:, :, None])
    times = times.flatten()
    

    if downsample_points < Data.shape[1]: #check if there is a need for downsampling
        tmp_data = np.zeros([Data.shape[0],downsample_points,Data.shape[2],1])
        tmp_times = np.zeros([1,downsample_points])


        for idx_for_downsample in range(Data.shape[0]):
            tmp_data[idx_for_downsample,:,:, :] = Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :]
        times = times[ids_downsampled[idx_for_downsample,:]]

        Data = tmp_data.copy()

    if verbose:
        print('Data.shape: ',Data.shape)
        print('times.shape: ',times.shape)

    # scaling_factor = to_np(Data).max()
    # args.scaling_factor = np.quantile(Data.flatten(), 0.99) #Data.max()
    Data = Data/scaling_factor
    Data_test = Data_test/scaling_factor

    Data = torch.Tensor(Data)
    Data_test = torch.Tensor(Data_test)

    #Original Dataset setup 
    Data_splitting_indices = Train_val_split(np.copy(index_np),validation_split) #Just the first 100 are used for training and validation
    Train_Data_indices = Data_splitting_indices.train_IDs()
    Val_Data_indices = Data_splitting_indices.val_IDs()

    if args.randomly_drop_n_last_frames is not None:
        frames_to_drop = np.random.randint(args.randomly_drop_n_last_frames, size=len(Val_Data_indices)+len(Train_Data_indices))
    elif args.drop_n_last_frames is not None:
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
    elif args.num_points_for_c is not None:
        args.drop_n_last_frames = Data.shape[1]-args.num_points_for_c
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames

    if verbose:
        print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
        print('Train_Data_indices: ',Train_Data_indices)
        print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
        print('Val_Data_indices: ',Val_Data_indices)

    Dataset_mine = Dynamics_Dataset3(Data,times,frames_to_drop)

    # For the sampler
    train_sampler = SubsetRandomSampler(Train_Data_indices)
    valid_sampler = SubsetRandomSampler(Val_Data_indices)

    dataloaders = {'train': torch.utils.data.DataLoader(Dataset_mine, sampler=train_sampler, batch_size = batch_size_segments,
                                                        num_workers=6, drop_last=False),
                   'val': torch.utils.data.DataLoader(Dataset_mine, sampler=valid_sampler, batch_size = batch_size_segments, 
                                                       num_workers=6, drop_last=False),
                  }
    
    return dataloaders

def create_dataloaders_toydata_eval(path_file, experiment_name, use_first_n_frames, batch_size_segments, validation_split, regularly_sampled, downsample_points,args):
    # This version returns the whole data as well. This will be used to compute the interpolation loss

    
    print('Loading ',os.path.join(path_file,experiment_name + ".p"))
    Data_dict = pickle.load(open(os.path.join(path_file,experiment_name + ".p"), "rb" )) #This data was saved in GPU. So transform it to CPU first
    print(Data_dict.keys())
    Data = Data_dict['Data_orig'][:,:use_first_n_frames,:, None]

    if args.add_noise_to_input_data>0:
        for idx_curve in range(Data.shape[0]):
            noise_0 = np.random.normal(0,args.add_noise_to_input_data,Data.shape[1])
            noise_1 = np.random.normal(0,args.add_noise_to_input_data,Data.shape[1])
            Data[idx_curve,:,0,0] = Data[idx_curve,:,0,0]+noise_0
            Data[idx_curve,:,1,0] = Data[idx_curve,:,1,0]+noise_1

    Data_orig = Data.copy()
    print('Data.shape: ',Data.shape)

    if downsample_points < Data.shape[1]: #check if there is a need for downsampling
        if regularly_sampled:
            # print('Data.shape: ',Data.shape)
            # print('args.downsample_points: ',args.downsample_points)
            if args.T_to_sample is not None:
                print('len(args.T_to_sample): ',len(args.T_to_sample))
                print('args.downsample_points: ',args.downsample_points)
                if args.downsample_points == len(args.T_to_sample): #In this case, just use the given coordinates and downsample anything
                    ids_downsampled = np.tile(args.T_to_sample,(Data.shape[0],1))
            else: 
                ids_downsampled = np.tile(np.linspace(0,Data.shape[1]-1,num=downsample_points, dtype=np.int64),(Data.shape[0],1))
        else: 
            random_idxs = np.zeros([Data.shape[0],9])
            for idx_tmp in range(Data.shape[0]):
                random_idxs[idx_tmp,:] = np.random.choice(np.arange(1, Data.shape[1],dtype=np.int64), size=downsample_points-1, replace=False)
            # print('random_idxs: ',random_idxs)

            ids_downsampled = np.concatenate((np.zeros([Data.shape[0],1],dtype=np.int8),random_idxs),axis=1).astype(int)
            # print('ids_downsampled: ',ids_downsampled)

            ids_downsampled = np.sort(ids_downsampled)
        print('ids_downsampled[0]: ',ids_downsampled[0])

    scaling_factor = np.quantile(np.abs(Data),0.98)
    print('scaling_factor: ',scaling_factor)

    # args.range_imshow = np.array([np.quantile(Data.flatten(), 0.4), np.quantile(Data.flatten(), 0.55)])#np.array([-0.25,0.05]) #
    # print('args.range_imshow: ',args.range_imshow)
    # args.fitted_pca = Data_dict['pca']
    # Data = to_np(Data[:,:4]) #This might be necessary in some cases. Not sure why some of these variables were saved as CUDA.

    train_val = 20000 # Number of frames for train and validation. The remaining will be for test
    n_steps = 3000 #number of iterations for training. default=3k epochs
    # segment_len = args.segment_len

    Data_test = Data[train_val:,:]
    # Data = Data[:args.use_first_n_frames,:] #Data[:train_val,:]

    n_points = Data.shape[1]
    extrapolation_points = Data.shape[1]

    # t_max = 1 #frames.shape[0]
    t_max=1
    t_min=0

    index_np = np.arange(0, len(Data), 1, dtype=int)
    index_np = np.hstack(index_np[:, None])
    times_np = np.linspace(t_min, t_max, num=n_points) #Original
    # times_np = np.linspace(t_min, t_max, num=args.segment_len)
    times_np = np.hstack([times_np[:, None]])
    # print('times_np: ',times_np)

    ###########################################################
    times = torch.from_numpy(times_np[:, :, None])#.to(args.device)
    times = times.flatten()
    times_orig = times.clone()
    
    print('times.shape: ',times.shape)
    print('times: ',times)

    if downsample_points < Data.shape[1]: #check if there is a need for downsampling
        tmp_data = np.zeros([Data.shape[0],downsample_points,Data.shape[2],1])
        tmp_times = np.zeros([1,downsample_points])
        print('tmp_data.shape: ',tmp_data.shape) #tmp_data.shape:  (100, 10, 2, 1)
        print('tmp_times.shape: ',tmp_times.shape) #tmp_times.shape:  (100, 10)


        for idx_for_downsample in range(Data.shape[0]):
            # print('idx_for_downsample: ',idx_for_downsample)
            # print('tmp_data[idx_for_downsample,:,:, :].shape: ',tmp_data[idx_for_downsample,:,:, :].shape)
            # print('Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape: ',Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape)
            # print('tmp_times[idx_for_downsample,:].shape: ',tmp_times[idx_for_downsample,:].shape)
            # print('times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape: ',times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape)
            tmp_data[idx_for_downsample,:,:, :] = Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :]
        times = times[ids_downsampled[idx_for_downsample,:]]

        Data = tmp_data.copy()
        # times = tmp_times.copy()

    # time_seq = times/t_max
    # print('time_seq: ',time_seq)
    print('Data.shape: ',Data.shape)
    print('times.shape: ',times.shape)
    print('Data_test.shape: ',Data_test.shape)

    # scaling_factor = to_np(Data).max()
    # args.scaling_factor = np.quantile(Data.flatten(), 0.99) #Data.max()
    Data = Data/scaling_factor
    Data_orig = Data_orig/scaling_factor
    Data_test = Data_test/scaling_factor

    # Data = torch.from_numpy(Data).to(args.device)
    Data = torch.Tensor(Data)#.double()
    Data_orig = torch.Tensor(Data_orig)
    Data_test = torch.Tensor(Data_test)#.double()
    # times = torch.Tensor(times)#.double()

    #Original Dataset setup 
    # Data_splitting_indices = Train_val_split3(np.copy(index_np),args.validation_split, args.segment_len,args.segment_window_factor) #Just the first 100 are used for training and validation
    # validation_split=0.3
    Data_splitting_indices = Train_val_split(np.copy(index_np),validation_split) #Just the first 100 are used for training and validation
    Train_Data_indices = Data_splitting_indices.train_IDs()
    Val_Data_indices = Data_splitting_indices.val_IDs()

    if args.randomly_drop_n_last_frames is not None:
        frames_to_drop = np.random.randint(args.randomly_drop_n_last_frames, size=len(Val_Data_indices)+len(Train_Data_indices))
    elif args.drop_n_last_frames is not None:
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
    elif args.num_points_for_c is not None:
        args.drop_n_last_frames = Data.shape[1]-args.num_points_for_c
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
    print('frames_to_drop.shape: ',frames_to_drop.shape)
    print('frames_to_drop: ',frames_to_drop)

    # frames_to_drop = np.random.randint(randomly_drop_n_last_frames+1, size=len(Data))
    print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
    print('Train_Data_indices: ',Train_Data_indices)
    print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
    print('Val_Data_indices: ',Val_Data_indices)
    # print('frames_to_drop [for train]: ',frames_to_drop[Train_Data_indices])
    # print('frames_to_drop [for val]: ',frames_to_drop[Val_Data_indices])
    # # #Define frames to drop
    # if args.randomly_drop_n_last_frames is not None:
    #     args.randomly_drop_n_last_frames = np.random.randint(args.randomly_drop_n_last_frames, size=len(Val_Data_indices)+len(Train_Data_indices))
    # print('args.randomly_drop_n_last_frames; ',args.randomly_drop_n_last_frames)

    # Dataset = Dynamics_Dataset2(Data,times,args.segment_len,args.segment_window_factor, frames_to_drop)
    Dataset_mine = Dynamics_Dataset4(Data,Data_orig,times,times_orig, frames_to_drop)
    # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
    # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

    # times_np_test = np.linspace(t_min, t_max, num=Data_test.shape[0])
    # times_np_test = np.hstack([times_np_test[:, None]])
    # times_test = torch.from_numpy(times_np_test[:, :, None])#.to(args.device)
    # times_test = times_test.flatten()
    # Dataset_all = Test_Dynamics_Dataset(Data,times)

    # For the sampler
    if args.mode=='train':
        train_sampler = SubsetRandomSampler(Train_Data_indices)
        valid_sampler = SubsetRandomSampler(Val_Data_indices)

        # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

        dataloaders = {'train': torch.utils.data.DataLoader(Dataset_mine, sampler=train_sampler, batch_size = batch_size_segments,
                                                            num_workers=args.num_workers, drop_last=False),
                       'val': torch.utils.data.DataLoader(Dataset_mine, sampler=valid_sampler, batch_size = batch_size_segments, 
                                                           num_workers=args.num_workers, drop_last=False),
                       # 'test': torch.utils.data.DataLoader(Dataset_all, batch_size = len(times),  num_workers=args.num_workers)
                      }
        # print('dataloaders: ',dataloaders)
    elif args.mode=='inference':
        valid_sampler = SequentialSampler(Val_Data_indices)
        num_workers = 6
        dataloaders   = {
            'val': DataLoader(Dataset_mine, batch_size=batch_size_segments, sampler=valid_sampler, num_workers=args.num_workers),
            }
    
    return dataloaders


def create_dataloaders_Navier_Stokes(Data, experiment_name, frame_size=64, margem_to_crop = [32,40,24,24], 
    fast_start=True, show_plots=False, batch_size_segments=8, segment_size = 50, behavior_variable=None, verbose=True, args=None):

    data = Data
    print('data.shape: ',data.shape)
    t_max=1
    t_min=0
    n_points = segment_size
    
    times = torch.linspace(t_min, t_max, n_points) #Original
    times_all = times
    
    if args.downsample_points < Data.shape[1]: #check if there is a need for downsampling
        if args.regularly_sampled:
            ids_downsampled = np.tile(np.linspace(0,Data.shape[1]-1,num=args.downsample_points, dtype=np.int64),(Data.shape[0],1))
        else: 
            random_idxs = np.zeros([Data.shape[0],9])
            for idx_tmp in range(Data.shape[0]):
                random_idxs[idx_tmp,:] = np.random.choice(np.arange(1, Data.shape[1],dtype=np.int64), size=args.downsample_points-1, replace=False)

            ids_downsampled = np.concatenate((np.zeros([Data.shape[0],1],dtype=np.int8),random_idxs),axis=1).astype(int)

            ids_downsampled = np.sort(ids_downsampled)
        print('ids_downsampled[0]: ',ids_downsampled[0])
    
    if args.downsample_points < Data.shape[1]: #check if there is a need for downsampling
        tmp_data = torch.zeros([Data.shape[0],args.downsample_points,Data.shape[2],Data.shape[3]])
        tmp_times = torch.zeros([1,args.downsample_points])
        print('tmp_data.shape: ',tmp_data.shape) #tmp_data.shape:  (100, 10, 2, 1)
        print('tmp_times.shape: ',tmp_times.shape) #tmp_times.shape:  (100, 10)


        for idx_for_downsample in range(Data.shape[0]):
            tmp_data[idx_for_downsample,:,:] = Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:],...]
        times = times[ids_downsampled[idx_for_downsample,:]]

        data = tmp_data.clone()
        
        print("Downsampled data: ",data.shape)
    
    
    class MyDataset(Dataset):
        def __init__(self, data, times, compression_factor):

            if compression_factor != 1:
                #print(data.shape)
                data = zoom(data, (1,1,compression_factor, compression_factor))
                #print(data.shape)

            self.data = data
            self.times = times.float()

        def __getitem__(self, index):
            x = self.data[index,...]
            t = self.times
            return x, t

        def __len__(self):
            return len(self.data)
    
    print('times.shape: ',times.shape)
    print('times: ',times)

    compression_factor=1
    range_imshow = np.array([np.quantile(data.flatten(), 0.01), np.quantile(data.flatten(), 0.99)])
    print('range_imshow: ',range_imshow)
    
    dataset = MyDataset(data, times, compression_factor)
    print('created dataset')
    dataset_size  = len(dataset)
    print('dataset_size: {}'.format(dataset_size))
    validation_split=args.validation_split

    # -- split dataset
    indices       = list(range(dataset_size))
    split         = int(np.floor(validation_split*dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    print('train_indices[:50]: {}'.format(train_indices[:50]))
    print('val_indices[:50]: {}'.format(val_indices[:50]))
    print('len(val_indices): ',len(val_indices))

    # -- create dataloaders
    if args.mode=='train':
        #Original
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(indices[0])

        
        num_workers = 6
        dataloaders   = {
            'train': DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers),
            'val': DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=num_workers),
            }
    elif args.mode=='inference':
        valid_sampler = SequentialSampler(val_indices)
        num_workers = 6
        dataloaders   = {
            'val': DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=num_workers),
            }

    other_vars ={'range_imshow': range_imshow}
    
    
    return dataloaders, other_vars
    
    