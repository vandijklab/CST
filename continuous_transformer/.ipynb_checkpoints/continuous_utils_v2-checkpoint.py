import os, random, torch, h5py
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SequentialSampler, SubsetRandomSampler
from skimage.transform import rescale, resize
import matplotlib.patches as patches


def print_nvidia_smi():
    bashCommand = "nvidia-smi"
    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output.decode('ascii'))
    
    
def patch_sampling(data, patch_sampling_cfg, frame_size, patch_size, current_epoch=None):
# def patch_sampling(data, hparams, frame_size, patch_size):
    sampling_type = patch_sampling_cfg["structure"]
#     print('hparams: ',hparams)

    print("frame_size:",frame_size)
    print("patch_size:",patch_size)
    
    if frame_size==patch_size:
        T, P_row, P_col = sampling_full_frames(data, patch_sampling_cfg, frame_size, patch_size)
    elif sampling_type == "random":
        T, P_row, P_col = sampling_random(data, patch_sampling_cfg, frame_size, patch_size)
    else: # sampling_type == "grid"
        if patch_sampling_cfg["mode"]=="inference":
            T, P_row, P_col = sampling_grid(data, patch_sampling_cfg, frame_size, patch_size, current_epoch)
        else:
            T, P_row, P_col = sampling_grid(data, patch_sampling_cfg, frame_size, patch_size)

    return sampling_segments(data, T, P_row, P_col, patch_sampling_cfg, frame_size, patch_size)

def sampling_full_frames(data, patch_sampling_cfg, frame_size, patch_size):
    # -- select T samples with replacement for patches
    num_patches = patch_sampling_cfg["num_patches"]
    if data.shape[0]> num_patches:
        rep_param = False
    else:
        rep_param = True
        
    T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]), num_patches, replace=rep_param)))
    P_row = torch.from_numpy(np.zeros_like(T))
    P_col = torch.from_numpy(np.zeros_like(T))
    
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
        
#     # Original
#     T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]), num_patches, replace=rep_param))) # Select T samples with 
#     P_row = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches)))
#     P_col = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches)))

##################################################################
#     # Sample random patches from the sequence, but get non-overlapping pathes for the last frame of the segment, to make sure we can make reconstruction
#     T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]-1), num_patches-n_pred, replace=rep_param))) # Select T samples with 
#     P_row = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
#     P_col = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
    
#     T_lastFrame = torch.tensor([data.shape[0]-1])
#     P_row_lastFrame = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
#     P_col_lastFrame = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
    
#     T_lastFrame,P_row_lastFrame,P_col_lastFrame = torch.meshgrid(T_lastFrame,P_row_lastFrame,P_col_lastFrame)
#     T_lastFrame,P_row_lastFrame,P_col_lastFrame = T_lastFrame.flatten(),P_row_lastFrame.flatten(),P_col_lastFrame.flatten()
    
#     T = torch.cat((T,T_lastFrame))
#     P_row = torch.cat((P_row,P_row_lastFrame))
#     P_col = torch.cat((P_col,P_col_lastFrame))

# ##################################################################
#     # Sample random patches from the sequence, but sample n_pred from the last frame only (no need to make whole frame reconstruction for this configuration)
#     T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]-1), num_patches-n_pred, replace=rep_param))) # Select T samples with 
#     P_row = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
#     P_col = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
    
#     T_lastFrame = torch.ones(n_pred,dtype=torch.int64)*(data.shape[0]-1)
#     P_row_lastFrame = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=n_pred)))
#     P_col_lastFrame = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=n_pred)))
    
#     T = torch.cat((T,T_lastFrame))
#     P_row = torch.cat((P_row,P_row_lastFrame))
#     P_col = torch.cat((P_col,P_col_lastFrame))


# ##################################################################
#     # Sample random patches from the sequence, but sample n_pred from the last frame only (no need to make whole frame reconstruction for this configuration)
#     # Also, adding implementation to work on batches
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
#     print("mode: ", mode)
#     print('current_epoch: ',current_epoch)
#     grid_rows = frame_size//patch_size
#     print('data.shape[0], num_patches: ',data.shape[0], num_patches)
    if data.shape[1]> num_patches:
        rep_param = False
    else:
        rep_param = True
    
#     # Gutavo's method
#     T = torch.from_numpy(np.repeat(np.sort(np.random.choice(np.arange(data.shape[0]), num_patches, replace=rep_param)), grid_rows**2)) # Select T samples with 
# #     T = torch.from_numpy(np.repeat(np.arange(grid_rows**2), num_patches))    
#     P_row = torch.from_numpy(np.tile(np.tile(np.arange(grid_rows) * patch_size, grid_rows), num_patches))
#     P_col = torch.from_numpy(np.tile(np.repeat(np.arange(grid_rows) * patch_size, grid_rows), num_patches))
# #     print('Gustavo:')
# #     print('T: ',T)
# #     print('P_row: ',P_row)
# #     print('P_col: ',P_col)
# #     print('T: {}, P_row: {}, P_col: {}'.format(T.shape, P_row.shape, P_col.shape))

#     # To sample a grid from the frame
#     T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]), num_patches, replace=rep_param))) # Select T samples with 
#     P_row = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
#     P_col = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
# #     T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Original, but potentially wrong
#     T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Needs to invert x and y to match actual 
#     T,P_row,P_col = T.flatten(),P_row.flatten(),P_col.flatten()

    if mode=="inference":
#         if current_epoch%2==0:
            # To sample a grid from the frame and make sure that last frame is always sampled, because we want to predict that one everytime
        T = torch.arange(0,data.shape[1]-1) # Select T samples with 
#         else:
#             T = torch.arange(1,data.shape[0]-1,step=2) # Select T samples with 
        T = torch.cat((T,torch.tensor([data.shape[1]-1]))) # Append the last frame manually 
#         P_row = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
#         P_col = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
        P_row = torch.linspace(0,frame_size["rows"]-patch_size-1,steps=int(frame_size["rows"]/patch_size),dtype=torch.int64)
        P_col = torch.linspace(0,frame_size["cols"]-patch_size-1,steps=int(frame_size["cols"]/patch_size),dtype=torch.int64)
    #     T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Original, but potentially wrong
        T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Needs to invert x and y to match actual 
        T,P_row,P_col = T.flatten(),P_row.flatten(),P_col.flatten()
            
        
    else: 
        # To sample a grid from the frame and make sure that last frame is always sampled, because we want to predict that one everytime
        T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[1]-1), num_patches-1, replace=rep_param))) # Select T samples with 
        T = torch.cat((T,torch.tensor([data.shape[1]-1]))) # Append the last frame manually 
        

        P_row = torch.linspace(0,frame_size["rows"]-patch_size-1,steps=int(frame_size["rows"]/patch_size),dtype=torch.int64)
        P_col = torch.linspace(0,frame_size["cols"]-patch_size-1,steps=int(frame_size["cols"]/patch_size),dtype=torch.int64)
    #     T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Original, but potentially wrong
        T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Needs to invert x and y to match actual 
        T,P_row,P_col = T.flatten(),P_row.flatten(),P_col.flatten()
    
#     print('Mine:')
#     print('T: ',T)
#     print('P_row: ',P_row)
#     print('P_col: ',P_col)
#     print('T: {}, P_row: {}, P_col: {}'.format(T.shape, P_row.shape, P_col.shape))
#     print(aaaa)
    
    return T, P_row, P_col

def sampling_segments(data, T, P_row, P_col, patch_sampling_cfg, frame_size, patch_size):
    verbose =False
    if verbose: 
        print('data.shape: ',data.shape)
        print('T: ',T)
        print('P_row: ',P_row)
        print('P_col: ',P_col)
        print('T: {}, P_row: {}, P_col: {}'.format(T.shape, P_row.shape, P_col.shape))

    num_patches = patch_sampling_cfg["num_patches"]
    num_frames = patch_sampling_cfg["num_frames"] # NUmber of frames in teh segment
    n_pred = patch_sampling_cfg["num_patches_to_hide"]
    batch_size_segments = data.shape[0]
#     print('num_patches: ',num_patches)
    
#     segm_frames = torch.zeros((num_patches, patch_size**2), dtype=torch.float32) # Using patches. This would work for patch=frame size

# This is hard coded for the grid... The long term solution should use the 'T' since it is already coming as a grid
#     segm_frames = torch.zeros((num_patches*int((frame_size/patch_size)**2),patch_size**2),dtype=torch.float32) # Using patches 
    
    #Orignal
#     segm_frames = torch.zeros((T.shape[0],patch_size**2),dtype=torch.float32) # Using patches 
    segm_frames = torch.zeros((batch_size_segments,T.shape[0],patch_size**2),dtype=torch.float32) # Using patches 
    
    if verbose: print('segm_frames.shape: ',segm_frames.shape)
    for idx1 in range(batch_size_segments):
        for idx in range(len(T)):
            if verbose: print('idx: {}, T[idx]: {}, P_row[idx]: {}, P_col[idx]: {}'.format(idx, T[idx], P_row[idx], P_col[idx]))
            segm_frames[idx1,idx, :] = data[idx1, T[idx], P_row[idx]:P_row[idx]+patch_size, P_col[idx]:P_col[idx]+patch_size].flatten()

#     # Original
#     cand_pos = np.arange(1, segm_frames.shape[0]-1) # Do not replace the first and last tokens
#     if frame_size == patch_size:
#         random.shuffle(cand_pos)
        
    # Mask just the last patches (ie, the patches of the last frames)
    # Original 
#     cand_pos = np.arange(segm_frames.shape[0]-n_pred, segm_frames.shape[0]) 
    cand_pos = np.arange(segm_frames.shape[1]-n_pred, segm_frames.shape[1]) 
    
    if verbose: print('cand_pos: ',cand_pos)
#     print(aaaa)

    tmp_masked_pos = []
    
#    #Original
#     tmp_masked_tokens = torch.empty(n_pred, segm_frames.shape[1]) # replace this constant by a variable

    tmp_masked_tokens = torch.empty(batch_size_segments, n_pred, segm_frames.shape[2]) 

    dict_special_tokens = {
        "MASK": 0.5
    }
    MASK_FRAME = torch.from_numpy(np.ones(patch_size**2)*dict_special_tokens["MASK"])
    
#     #Original 
#     for pos, idx_pos in zip(cand_pos[:n_pred], range(n_pred)):
#         tmp_masked_tokens[idx_pos, :] = segm_frames[pos, :]    
#         tmp_masked_pos.append(pos)
#         if random.random() < 1: # 0.8
#             segm_frames[pos, :] = MASK_FRAME

    for idx1 in range(batch_size_segments):
        for pos, idx_pos in zip(cand_pos[:n_pred], range(n_pred)):
            tmp_masked_tokens[idx1, idx_pos, :] = segm_frames[idx1, pos, :]    
            tmp_masked_pos.append(pos)
    #         if random.random() < 1: # 0.8
            segm_frames[idx1, pos, :] = MASK_FRAME

#     # Original
#     masked_pos = torch.zeros((1, n_pred), dtype=torch.int64)
#     masked_pos[0, :len(tmp_masked_pos)] = torch.tensor(np.array(tmp_masked_pos, dtype=np.int64))# dtype=torch.int64)
#     masked_pos = torch.from_numpy(np.array(masked_pos))
#     masked_tokens = []
#     masked_tokens.append(tmp_masked_tokens)
#     masked_tokens = torch.stack(masked_tokens, dim=0)
#     masked_tokens = torch.reshape(
#         masked_tokens,
#         (masked_tokens.shape[0]*masked_tokens.shape[1], masked_tokens.shape[2])
#     )
#     masked_tokens = masked_tokens.squeeze().type_as(data)

#     masked_pos = torch.zeros((batch_size_segments, n_pred), dtype=torch.int64)
    masked_pos = torch.tensor(np.array(tmp_masked_pos, dtype=np.int64)).reshape(batch_size_segments,n_pred)
    masked_pos = torch.from_numpy(np.array(masked_pos))
    
    if verbose: 
        print('len(tmp_masked_pos): ',len(tmp_masked_pos))
        print('tmp_masked_pos: ',tmp_masked_pos)
        print('tmp_masked_tokens.shape: ',tmp_masked_tokens.shape)
        print('masked_pos.shape: ',masked_pos.shape)
        print('masked_pos: ',masked_pos)
#     masked_tokens = []
#     masked_tokens.append(tmp_masked_tokens)
#     masked_tokens = torch.stack(masked_tokens, dim=0)
#     masked_tokens = torch.reshape(
#         masked_tokens,
#         (masked_tokens.shape[0]*masked_tokens.shape[1], masked_tokens.shape[2])
#     )
#     masked_tokens = tmp_masked_tokens.reshape(batch_size_segments,num_patches, patch_size, patch_size) #What was using for test 402
    masked_tokens = tmp_masked_tokens.reshape(batch_size_segments,n_pred, patch_size, patch_size)
    if verbose: print('masked_tokens.shape: ', masked_tokens.shape)
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
#     ax.plot(train_loss_epoch)
    ax.plot(val_loss.detach().cpu())
    ax.set_yscale('log')
    ax.legend(['Validation'], fontsize=15)
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15) 
    ax.set_title('Total losses (Contrast labels)', fontsize=15)
    ax.set_aspect('auto')

#     ax2.plot(r2_landmarks_train)
#     ax2.plot(r2_landmarks_val)
#     ax2.plot(r2_reg_train)
    ax2.plot(val_r2)
    #             ax2.legend(['Rsquared'], fontsize=15)
    ax2.legend(['Val_LM'], fontsize=15)
    ax2.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('R^2', fontsize=15) 
    ax2.set_title('R^2', fontsize=15)
    ax2.set_aspect('auto')
        
    plt.savefig(os.path.join(save_path, "plots", f"training-curves-epoch-{epoch:05d}.png"))
    plt.close('all')
    
def plot_predictions(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path):
    # block displaying plots during training
    plt.ioff()

    patches_to_plot = 32 
    if logits_lm.dim()==1:
        logits_lm = logits_lm[None,:]
        masked_tokens = masked_tokens[None,:]
        
#     if  masked_tokens.dim()==1: # In this case, we are plotting the whole frame instead of patches
#         fig, ax = plt.subplots(1,2, figsize=(10,5))
#         idx_tmp =0
#         tmp_maskpos = masked_pos.detach().cpu().numpy().flatten()
#         ax[0].imshow(masked_tokens.reshape(patch_size,patch_size).detach().cpu(), vmin=-0.5, vmax=0.5)
#         ax[0].set_title(f"masked_tokens[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}, x: {P_row[tmp_maskpos[idx_tmp]]}, y: {P_col[tmp_maskpos[idx_tmp]]}")
#         ax[0].set_xticks([])
#         ax[0].set_yticks([])

#         ax[1].imshow(logits_lm.reshape(patch_size,patch_size).detach().cpu(), vmin=-0.5, vmax=0.5)
#         r2_sq = (np.corrcoef(logits_lm.detach().cpu().numpy().flatten(), masked_tokens.detach().cpu().numpy().flatten(), rowvar=False)[0,1])**2
#         ax[1].set_title(f"logits_lm[{idx_tmp}], r2: {r2_sq:.2f}")
#         ax[1].set_xticks([])
#         ax[1].set_yticks([])
            
#     else:
    tmp_size = np.minimum(masked_tokens.shape[1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
    if tmp_size > 2:
        n_ = int(np.ceil(np.sqrt(2*tmp_size))) # 2x because I want to plot masked patch and prediction 
        fig, ax = plt.subplots(n_,n_, figsize=(50,50))
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5))

    ax = ax.ravel()
    c=0
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
#     for idx_tmp in range(masked_tokens.shape[1]-1, masked_tokens.shape[1]-tmp_size-1,-1):
    list_idx = np.sort(np.random.choice(np.arange(masked_tokens.shape[1]), tmp_size, replace=True))
    for idx_tmp in list_idx:
        ax[c].imshow(masked_tokens[0,idx_tmp,:].reshape(patch_size,patch_size).detach().cpu(), vmin=-0.5, vmax=0.5)
        ax[c].set_title(f"masked_tokens[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}, row: {P_row[tmp_maskpos[idx_tmp]]}, col: {P_col[tmp_maskpos[idx_tmp]]}")
        ax[c].set_xticks([])
        ax[c].set_yticks([])
        c+=1
#         print('logits_lm.shape: ',logits_lm.shape)
        A = logits_lm[0,idx_tmp,:].detach().cpu().numpy().flatten()
        B = masked_tokens[0,idx_tmp,:].detach().cpu().numpy().flatten()
        
#         # Check if they are all the same value. If yes, add a small jitter just to be able to compute R2
#         if len(np.unique(A)):
#             A=A+np.random.rand(len(A))*10e-10
#         if len(np.unique(B)):
#             B=B+np.random.rand(len(B))*10e-10
            
        r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2


        ax[c].imshow(logits_lm[0,idx_tmp,:].reshape(patch_size,patch_size).detach().cpu(), vmin=-0.5, vmax=0.5)
        ax[c].set_title(f"logits_lm[{idx_tmp}], r2: {r2_sq:.2f}")
        ax[c].set_xticks([])
        ax[c].set_yticks([])
        c+=1

    plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-val-loss-{val_loss:.6f}-val-r2-{val_r2:.2f}.png"))
    plt.close('all')
    
def plot_whole_frame(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,frame_size,
                     epoch, val_loss, val_r2, save_path):
    
    fig,ax = plt.subplots(1,2, facecolor='w',figsize=(10,6))
    
    tmp_img1 = torch.zeros(frame_size["rows"],frame_size["cols"])
    count=-int((frame_size["rows"]/patch_size)*(frame_size["cols"]/patch_size))
    for idx_patch1 in torch.linspace(0,frame_size["rows"]-patch_size,steps=int(frame_size["rows"]/patch_size),dtype=torch.int64):
        for idx_patch2 in torch.linspace(0,frame_size["cols"]-patch_size,steps=int(frame_size["cols"]/patch_size), dtype=torch.int64):
            tmp_img1[idx_patch1:idx_patch1+patch_size, idx_patch2:idx_patch2+patch_size] = logits_lm[0,count,:].reshape(patch_size,patch_size)
            count+=1

    ax[1].imshow(tmp_img1,vmin=-0.5, vmax=0.5)
    count=-int((frame_size["rows"]/patch_size)*(frame_size["cols"]/patch_size))
#     for idx_patch1 in torch.linspace(0,frame_size-patch_size,steps=int(frame_size/patch_size),dtype=torch.int64):
#         for idx_patch2 in torch.linspace(0,frame_size-patch_size,steps=int(frame_size/patch_size), dtype=torch.int64):
#             ax[1].annotate(str(-count)+',row:' +str(idx_patch1.numpy())+', col:'+str(idx_patch2.numpy()) ,xy=(idx_patch2,idx_patch1), xytext=(idx_patch2,idx_patch1)) # It is inverted because x is col and y is row
#             count+=1
# #                     ax[1].set_title('Reconstructed (Pred Label: {:.4f})'.format(logits[0].cpu().numpy()))
    ax[1].set_title('Reconstructed (Frame: {})'.format(T[-1]))
    
    tmp_img = torch.zeros(frame_size["rows"],frame_size["cols"])
    count= -int((frame_size["rows"]/patch_size)*(frame_size["cols"]/patch_size))
    for idx_patch1 in torch.linspace(0,frame_size["rows"]-patch_size,steps=int(frame_size["rows"]/patch_size),dtype=torch.int64):
        for idx_patch2 in torch.linspace(0,frame_size["cols"]-patch_size,steps=int(frame_size["cols"]/patch_size), dtype=torch.int64):
            tmp_img[idx_patch1:idx_patch1+patch_size, idx_patch2:idx_patch2+patch_size] = masked_tokens[0,count,:].reshape(patch_size,patch_size)
            count+=1
#                     ax[0].imshow(masked_tokens[0,:].reshape(patch_size,patch_size),vmin=-0.5, vmax=0.5)
    ax[0].imshow(tmp_img, vmin=-0.5, vmax=0.5)
    ax[0].set_title('Original')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    r_square_frames = (np.corrcoef(tmp_img1.flatten(), 
                                   tmp_img.flatten(), rowvar=False)[0,1])**2
    fig.suptitle('R2 = {:.4f}'.format(r_square_frames))
#     plt.show()

    plt.savefig(os.path.join(save_path, "plots", f"whole_frame-epoch-{epoch:05d}-val-loss-{val_loss:.6f}-val-r2-{r_square_frames:.3f}.png"))
#     plt.close('all')
    
    reconstrcuted_frame = tmp_img1.clone()
    
    return reconstrcuted_frame



def create_dataloaders(path_file, experiment_name, frame_size=64, margem_to_crop = [32,40,24,24], fast_start=True, show_plots=False, batch_size_segments=10, segment_size = 32, verbose=True):
    # everything from bellow in one cell
    # Defining the dataset to be used
    dataSource = 'mouse' # '2dWave' is a toy dataset
    use_SimCLR=False
    orig_video=True
    window = 50 # Number of frames in the MMD window 
    loadCNMF = False # if true, load the factorization results from the cNMF method
    add_time_to_origVideo = True # If true and orig_video==True, then add time as channels in the original video.
    label_last_frame = False # if True, the last frame of the MMD window is used as the label for the window (the idea of make future predictions based on the window information)
    use_SetTransformer = False # if True, uses the Set Transformer to encode the windows

    if dataSource == 'mouse': # To load the Ca imaging video
#         path_file = os.path.join(path_file, experiment_name)
        output_path = os.path.join(path_file,experiment_name, "output")
        output_preprocessed = os.path.join(path_file,experiment_name, "output_preprocessed")
        # experiment = '11232019_grabAM07_vis_stim'
        fullpath = os.path.join(path_file,experiment_name, "final_dFoF.mat")
#         print(f"stim full path: {fullpath}")

    #     path_file = "/home/antonio2/data/2pmeso/imaging_with_575_excitation/Cardin/GRABS_Data_March/Analyzed_SVDMethodPatch14/DualMice"
    #     output_path = "/home/antonio2/data/2pmeso/output"
    #     output_preprocessed = '/home/antonio2/data/2pmeso/preprocessed'
    #     experiment = '11222019_grabAM05_spont'
    #     fullpath = os.path.join(path_file,experiment,"final_dFoF.mat")
    #     #print(fullpath)
    #     fast_start=True #True if we just want to load the saved variables (much faster)
    #     orig_video = False # True to work with original video. False to work with WFT
    #     single_freq = False # True to perform classification using single bandwidths from the fourier for test
    #     add_time_to_origVideo = True # If true, add time as channels in the original video.
    #     use_SimCLR=False

#     if dataSource == '2dWave': #To load the 2D wave video
#         path_file = "/gpfs/ysm/home/ahf38/Documents/2p_meso/data/AnalyzedData/grabAM07/imaging_with_575_excitation" 
#         output_path = "/gpfs/ysm/home/ahf38/project/2p_meso"
#         experiment = '11232019_grabAM07_vis_stim'
#         filename = 'test_diffFreqs_freq0.008_nSources2.mp4'
#         video_name = filename[:-4]
#         #print('video_name: {}'.format(video_name))
#         fullpath = os.path.join(output_path,experiment_name,"output", filename)
#         path_saveplot = os.path.join(output_path,experiment_name,"output")
#         #print(fullpath)

    if not fast_start:
        if verbose: print('Loading ',fullpath)
        f2 = h5py.File(fullpath,'r')
        #print(f2.keys())
        x = f2["dFoF"]
        #print(list(x))
        video = np.array(x['green'])
        if verbose: print('video loaded shape: ',video.shape)
            
        frame_width = int(np.sqrt(video.shape[1]))

        video = np.transpose(video.reshape(video.shape[0],frame_width, frame_width),(0,2,1)) #Turn video in frames, width, height
        if frame_size != video.shape[2]: 
            video = resize(video.T, (frame_size, frame_size, video.shape[0])).T
            if verbose: print('new video shape: ',video.shape)
                
#         data3 = video.reshape(1,video.shape[0],video.shape[1],video.shape[2]) # Original
        data3 = video.reshape(video.shape[0],video.shape[1],video.shape[2]) # Original
#         data3 = np.transpose(data3,(0,1,3,2))
        if verbose: print('video original shape: ',data3.shape)
        
        data3 = data3[:,margem_to_crop[0]:video.shape[1]-margem_to_crop[1],margem_to_crop[2]:video.shape[2]-margem_to_crop[3]]
        if verbose: print('video after crop shape: ',data3.shape)
        data3[np.isnan(data3)] = 0
        
        
                
#         video2 = np.resize(video[:,:,:], ((video.shape[0], frame_size, frame_size)))
    if not fast_start and show_plots:
        fig,ax = plt.subplots()
        ax.imshow(video[1000,:,:], cmap='viridis_r')
        ax.set_title('Sample frame: 1000')
        plt.show()

        # #print('video: ',video)
        #print('data3.shape: ',data3.shape)
        #print('data3.min(): ',data3.min())
        fig,ax = plt.subplots()
        ax.imshow(data3[1000,:,:], cmap='viridis_r')
        ax.set_title('Replacing NaN by zeros')
        
        # Create a Rectangle patch
        rect = patches.Rectangle((20, 40), 50, 50, linewidth=1,
                                 edgecolor='r', facecolor="none")

        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.show()
        print('Rectangle for reference: xy:{},{}, width: {}, height: {}'.format(20,40,50,50))
        
        fig,ax = plt.subplots()
#         ax.imshow(data3[0,1000,40:40+50, 20:20+50], cmap='viridis_r')
        ax.imshow(data3[1000,40:40+50, 20:20+50], cmap='viridis_r')
        ax.set_title('data3[0,1000,40:40+50, 20:20+50]')
        plt.show()

#         np.save(output_preprocessed + "/video_orig.npy", data3)
    '''
    Loading the dataset (fast start mode)
    '''
    if dataSource == 'mouse':
        if fast_start==True:
            if orig_video==False:
                data3 = np.load(os.path.join(output_path,experiment_name,"output/video_WFT.npy"),allow_pickle=True)
            else:
                data3 = np.load(os.path.join(path_file,experiment_name,"output/video_orig.npy"),allow_pickle=True)
#             vis_stim_all = np.load(os.path.join(path_file,experiment_name,"output/vis_stim_all.npy"))
#             time = np.load(os.path.join(path_file,experiment_name,"output/time.npy"))
#             bins = np.load(os.path.join(path_file,experiment_name,"output/bins.npy"))
#             freqs = np.load(os.path.join(path_file,experiment_name,"output/freqs.npy"))
            #print('data3.shape: {}'.format(data3.shape))
            #print(f"vis_stim_all.shape: {vis_stim_all.shape}")
            #print('bins.shape: {}'.format(bins.shape))
            #print('freqs.shape: {}'.format(freqs.shape))


    if dataSource == '2dWave':
        #print('fast_start: {}, orig_video: {}'.format(fast_start, orig_video))
        if fast_start==True and orig_video==False:
            P3 = np.load(os.path.join(output_path,experiment_name,"output/freq_decomp_" + video_name + "64.npy"),allow_pickle=True)
        #     vis_stim_all = np.load(os.path.join(output_path,experiment,"output/vis_stim_all.npy"))
        #     time = np.load(os.path.join(output_path,experiment,"output/time.npy"))
            bins = np.load(os.path.join(output_path,experiment_name,"output/bins_" + video_name + "64.npy"))
            freqs = np.load(os.path.join(output_path,experiment_name,"output/freqs_" + video_name + "64.npy"))
            #print('P3.shape: {}'.format(P3.shape))
            #print('bins.shape: {}'.format(bins.shape))
            #print('freqs.shape: {}'.format(freqs.shape))
    '''
    Loading behavior variables for the mouse dataset
    '''

    if dataSource == 'mouse':
        from scipy.io import loadmat
        spike2_vars = loadmat(os.path.join(path_file,experiment_name,"smrx_signals_v3.mat"))

        vis_stim = spike2_vars['timestamps']['contrasts_bin_100'][0][0] * 100
        vis_stim2 = spike2_vars['timestamps']['contrasts_bin_50'][0][0] *50 #This is already the visual stim matching frames
        vis_stim3 = spike2_vars['timestamps']['contrasts_bin_20'][0][0] * 20
        vis_stim4 = spike2_vars['timestamps']['contrasts_bin_10'][0][0] * 10
        vis_stim5 = spike2_vars['timestamps']['contrasts_bin_5'][0][0] * 5
        vis_stim6 = spike2_vars['timestamps']['contrasts_bin_2'][0][0] * 2
        vis_stim_all = vis_stim + vis_stim2 + vis_stim3 + vis_stim4 + vis_stim5 + vis_stim6
        print('vis_stim_all.shape: ',vis_stim_all.shape)
        time = spike2_vars['timestamps']['timaging'][0][0]
        print('time.shape: ',time.shape)
        fs_imaging = round(time[1][0]-time[0][0],2)
        print('fs_imaging: ',fs_imaging)

        if show_plots:
            fig, ax = plt.subplots(1,1,figsize=(16, 4), dpi= 100, facecolor='w', edgecolor='k')
            fig.suptitle('Stimulus')
            tmp1 = vis_stim_all[:1000]
            tmp2 = time[:1000]
            ax.plot(time[:1000], vis_stim_all[:1000])
            ax.set_xticks(np.arange(60, max(time[:1000])+1, 5.0))

        # Behavior variables
        allwhellon = spike2_vars['timing']['allwheelon'][0][0]
        allwhelloff = spike2_vars['timing']['allwheeloff'][0][0]
        allwheel = np.concatenate((allwhellon,allwhelloff), axis=1)

        allwheel_analog = spike2_vars['channels_data']['wheelspeed'][0][0].squeeze()
        allwheel_analog_time = np.arange(0,allwheel_analog.shape[0])
        allwheel_analog_time = allwheel_analog_time/5000 #Sampled at 5kHz
        print('[1] allwheel_analog_time.shape: {}'.format(allwheel_analog_time.shape))
        fs_wheel = allwheel_analog_time[1]-allwheel_analog_time[0]
        print('[1] fs_wheel = ', fs_wheel)
        ratio_imaging_wheel = int(fs_imaging/fs_wheel)
        print('ratio_imaging_wheel: ',ratio_imaging_wheel)
        allwheel_analog = allwheel_analog[::ratio_imaging_wheel]
        allwheel_analog_time = allwheel_analog_time[::ratio_imaging_wheel]
        allwheel_analog = allwheel_analog[np.where((allwheel_analog_time>=time[0]) & (allwheel_analog_time<=time[-1]))]
        allwheel_analog_time = allwheel_analog_time[np.where((allwheel_analog_time>=time[0]) & (allwheel_analog_time<=time[-1]))]
        print('allwheel_analog.shape: {}'.format(allwheel_analog.shape))
        print('allwheel_analog_time.shape: {}'.format(allwheel_analog_time.shape))
        print('allwheel_analog_time: {},{}',allwheel_analog_time[0], allwheel_analog_time[-1])
        print('time: ',time[0],time[-1])
        allwheel_time = np.arange(time[0],time[-1],0.1)
        allwheel_tmp = np.zeros_like(allwheel_time)

        for i in range(0,len(allwheel)):
            allwheel_tmp[(allwheel_time>allwheel[i,0]) & (allwheel_time<allwheel[i,1])]=1
        allwheel = allwheel_tmp[:]
        if show_plots:
            fig, (ax,ax1) = plt.subplots(2,1,figsize=(16, 4), dpi= 100, facecolor='w', edgecolor='k')
            fig.suptitle('Wheel')
            ax.plot(allwheel_time[:1000], allwheel[:1000])
            print('allwheel.shape: {}'.format(allwheel.shape))

            ax1.plot(allwheel_analog_time[:1000], allwheel_analog[:1000])
            ax.set_xticks(np.arange(60, max(time[:1000])+1, 5.0))

        #Make sure both arrays have the same size
        if np.abs(time[0]-allwheel_analog_time[0])<np.abs(time[-1]-allwheel_analog_time[-1]):
            time = time[:len(allwheel_analog_time)]
            vis_stim_all = vis_stim_all[:len(allwheel_analog_time)]
#             data3 =  data3[:,:len(allwheel_analog_time),:,:]
            data3 =  data3[:len(allwheel_analog_time),:,:]
        else:
            time = time[len(allwheel_analog_time):]   
            vis_stim_all = vis_stim_all[len(allwheel_analog_time):]
#             data3 =  data3[:,len(allwheel_analog_time):,:,:]
            data3 =  data3[len(allwheel_analog_time):,:,:]

        print('allwheel_analog_time: {},{}, {}'.format(allwheel_analog_time[0], allwheel_analog_time[-1],allwheel_analog_time.shape))
        print('time: {},{}, {}'.format(time[0],time[-1],time.shape))
        print('data3.shape: ',data3.shape)
        
#     # Replace every frame in the sequence by the mean frame
#     mean_frame = np.nanmean(data3, axis=1).squeeze()
#     print('\nmean_frame.shape: ',mean_frame.shape)
#     data3 = np.repeat(mean_frame[None,...],data3.shape[1],axis=0)
#     del mean_frame
#     data3 = data3[None,:]
#     print('[after replace by mean frame] data3.shape: ',data3.shape, data3.min(), data3.max())
        
#     '''
#     Matching sampling of behavior variables and video
#     '''

#     if dataSource == 'mouse':
#         #print(bins.shape)
#         #print(vis_stim_all.shape)
#         #print(time.shape)
#         #print('time[0]: {}, time[-1]: {}'.format(time[0], time[-1]))
#         time2 = time - time[0]
#         #print('time2: {}'.format(time2))
#         #print(bins)

#         bin_idx = []
#         for i in bins:
#             idx = np.argmin(abs(i-time2[:]))
#             bin_idx.append(idx)
#         #print('len(bin_idx): {}'.format(len(bin_idx)))
#         time_down = time2[bin_idx[:]]
#         #print(time_down[:10])

#     if dataSource == '2dWave':
#         time = np.arange(0,duration,1/fps)
#         #print('bins.shape: {}'.format(bins.shape))
#         #print('time.shape: {}'.format(time.shape))
#         #print('time[0]: {}, time[-1]: {}'.format(time[0], time[-1]))
#         time2 = time - time[0]
#         #print('time2: {}'.format(time2))
#         #print('bins: {}'.format(bins))

#         bin_idx = []
#         for i in bins:
#             idx = np.argmin(abs(i-time2[:]))
#             bin_idx.append(idx)
#         #print('len(bin_idx): {}'.format(len(bin_idx)))
#         time_down = time2[bin_idx[:]]


    '''
    Normalization
    '''

    if not loadCNMF:
        if use_SimCLR==True or use_SetTransformer:
            data_normalized = np.zeros((data3.shape[1], data3.shape[0], data3.shape[2], data3.shape[3]))

            for i in range(0, data_normalized.shape[0]): #Replace this by permute(1,0,2,3). Much simpler
                data_normalized[i,:,:,:] = data3[:,i,:,:]
            #print('data_normalized.shape: {}'.format(data_normalized.shape))
            # dataset = MyDataset(data, labels)

        else: #Use this option to transform the 4D to 2D 
            if orig_video:
                #print('using orig_video')
                orig_shape = data3.shape
#                 data3[0,:,:,:] = data3[0,:,:,:]/np.nanstd(data3[0,:,:,:])  
#                 data3[0,:,:,:] = ((data3[0,:,:,:] - np.nanmin(data3[0,:,:,:]))/(np.nanmax(data3[0,:,:,:]) - np.nanmin(data3[0,:,:,:])))*2-1
                data3[:,:,:] = data3[:,:,:]/np.nanstd(data3[:,:,:])  
                data3[:,:,:] = ((data3[:,:,:] - np.nanmin(data3[:,:,:]))/(np.nanmax(data3[:,:,:]) - np.nanmin(data3[:,:,:])))*2-1

#                 data_normalized = np.zeros((data3.shape[1], data3.shape[0] * data3.shape[2] * data3.shape[3]))
#                 for i in range(0, data_normalized.shape[0]): 
#                     data_normalized[i,:] = data3[:,i,:,:].reshape(1,data3.shape[0] * data3.shape[2] * data3.shape[3]) #Flattening everything

#                 data_normalized = data3.copy()
            else:
                data_normalized = np.zeros((data3.shape[1], data3.shape[0] * data3.shape[2] * data3.shape[3]))
                for i in range(0, data_normalized.shape[0]): 
                    data_normalized[i,:] = data3[:,i,:,:].reshape(1,data3.shape[0] * data3.shape[2] * data3.shape[3]) #Flattening everything
            if verbose: print('data3.shape: {}'.format(data3.shape))
                
    
    
    '''
    Normalize labels as well
    '''
    #     labels_tmp = allwheel_analog[bin_idx[:]]
    # labels = vis_stim_all[bin_idx[:]]
#     del data3
    labels1 = vis_stim_all
    labels2 = allwheel_analog
    labels1 = np.sqrt(labels1)
    # labels = allwheel[bin_idx[:]]
#      labels = ((labels - labels.min())/(labels.max() - labels.min()))*2-1
    labels1 = ((labels1 - labels1.min())/(labels1.max() - labels1.min()))*2-1
    
#     labels = {"vis_stim": labels1, "analog_wheel": labels2}
    # data_normalized = ((data_normalized - data_normalized.min())/(data_normalized.max() - data_normalized.min()))*2-1
    print('data3.shape: {}'.format(data3.shape))
    print('data3.min: {}'.format(data3.min()))
    print('data3.max: {}'.format(data3.max()))
    print('allwheel.shape: {}'.format(allwheel.shape))
    # print('vis_stim_all[bin_idx[:]].shape: {}'.format(vis_stim_all[bin_idx[:]].shape))
    print('vis_stim_all.shape: {}'.format(vis_stim_all.shape))
    
    if show_plots:
        print('\nplotting frame sample...')
        fig, (ax2, ax3) = plt.subplots(2,1,figsize=(15, 8), dpi= 80, facecolor='w', edgecolor='k')
        if orig_video:
#             print('np.nanmax(data_normalized,axis=1),np.nanmin(data_normalized,axis=1): ',np.nanmax(data_normalized,axis=1),np.nanmin(data_normalized,axis=1))
#             #Original
#             ax2.plot(np.nanmax(data_normalized,axis=1))
#             ax3.plot(np.nanmin(data_normalized,axis=1))
            
            ax2.plot(data3[:,:,:].max(axis=(1,2)))
            ax3.plot(data3[:,:,:].min(axis=(1,2)))

        else:
            ax2.plot(data3[:,:,:,:].max(axis=(0,2,3)))
            ax3.plot(data3[:,:,:,:].min(axis=(0,2,3)))

### Original dataset function #####
#     class MyDataset(Dataset):
#         def __init__(self, data, target, compression_factor):

#             if compression_factor != 1:
#                 #print(data.shape)
#                 data = zoom(data, (1,1,compression_factor, compression_factor))
#                 #print(data.shape)

#             self.data = torch.from_numpy(data).float()
#             self.target = torch.from_numpy(target).float()

#         def __getitem__(self, index):
#             print('index [MyDataSet]: ', index)
#             x = self.data[index]
#             y = self.target[index]
#             return x, y

#         def __len__(self):
#             return len(self.data)

    class MyDataset(Dataset):
        def __init__(self, data, target, compression_factor, segment_size):

            if compression_factor != 1:
                #print(data.shape)
                data = zoom(data, (1,1,compression_factor, compression_factor))
                #print(data.shape)

            self.data = torch.from_numpy(data).float()
            self.target = torch.from_numpy(target).float()
            self.segment_size = segment_size

        def __getitem__(self, index):
#             print('index [MyDataSet]: ', index)
            x = self.data[index:index+self.segment_size]
            y = self.target[index:index+self.segment_size]
            return x, y

        def __len__(self):
            return len(self.data)

        
    class MyDataset_test(Dataset):
        def __init__(self, data, target1, target2, compression_factor, segment_size):

            if compression_factor != 1:
                #print(data.shape)
                data = zoom(data, (1,1,compression_factor, compression_factor))
                #print(data.shape)

            self.data = torch.from_numpy(data).float()
            self.target1 = torch.from_numpy(target1).float()
            self.target2 = torch.from_numpy(target2).float()
            self.segment_size = segment_size

        def __getitem__(self, index):
#             print('index [MyDataSet]: ', index)
#             x = self.data[1000:2000]
#             y1 = self.target1[1000:2000]
#             y2 = self.target2[1000:2000]

            x = self.data[:10000]
            y1 = self.target1[:10000]
            y2 = self.target2[:10000]
            return x, y1, y2

        def __len__(self):
            return len(self.data)
    #print('max_labels {}, min {}'.format(np.amax(labels),np.amin(labels)))


    if orig_video:
        print('\ncopy data_normalized')
    #     data2 = data3[:,bin_idx[:],:,:] #For the 18D version, this operation is done in the concatenation step. Since there is no concat for orig_video, it has to be done here
        data = data3.copy() #For the 18D version, this operation is done in the concatenation step. Since there is no concat for orig_video, it has to be done here
        del data3
        print('finished copying')
    #     data = data_normalized[:,:] #For the 18D version, this operation is done in the concatenation step. Since there is no concat for orig_video, it has to be done here
    else:
        #print('Just copying data3')
        data2 = data3.copy()  
        del data3
        # data = data2.reshape(data2.shape[1], data2.shape[0], data2.shape[2], data2.shape[3]) #data2.shape: (17, 20477, 64, 64)
        data = np.zeros((data2.shape[1], data2.shape[0], data2.shape[2], data2.shape[3]))

        for i in range(0, data.shape[0]): #Replace this by permute(1,0,2,3). Much simpler
            data[i,:,:,:] = data2[:,i,:,:]
        del data2


    compression_factor=1
    dataset = MyDataset(data, labels1, compression_factor, segment_size)
    print('created dataset')
    dataset_size  = len(dataset)
    #print('dataset_size: {}'.format(dataset_size))
    validation_split=0.3

    # Number of frames in the sequence (in this case, same as number of tokens). Maybe I can make this number much bigger, like 4 times bigger, and then do the batches of batches...
    # For example, when classifying, I can test if the first and the second chunk are sequence vs the first and third
    # big_batch_size=300 # Number of frames in the minibatch [initially]
    #batch_size=32 #256 #How many frames are actually going to be selected
    #REMINDER: The batch size still needs to be adjusted with the same value in train_mrpc.json and bert_base.json files

    # -- split dataset
    indices       = list(range(dataset_size))
    split         = int(np.floor(validation_split*dataset_size))
    #print('train/val split: {}'.format(split))
    # np.random.shuffle(indices) # Randomizing the indices is not a good idea if you want to model the sequence
    train_indices, val_indices = indices[split:], indices[:split]
    train_indices, val_indices = train_indices[:-segment_size], val_indices[:-segment_size] #remove the indices at the end, since we want continuous windows of length segment_size
    print('train_indices[:50]: {}'.format(train_indices[:50]))

    # -- create dataloaders
    #Original
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(indices[0])

#     # Use sequential instead of random to make sure things are still in the right order
#     train_sampler = SequentialSampler(train_indices)
#     valid_sampler = SequentialSampler(val_indices)

    # # To make batch of batches, in which the minibatches are sequential
    # train_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(train_indices),batch_size=3, drop_last=True)
    # valid_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(val_indices),batch_size=3, drop_last=True)
    
    dataset_test = MyDataset_test(data, labels1, labels2, compression_factor, segment_size)
    
    num_workers = 6
    dataloaders   = {
        'train': DataLoader(dataset, batch_size=batch_size_segments, sampler=train_sampler, num_workers=num_workers),
        'val': DataLoader(dataset, batch_size=batch_size_segments, sampler=valid_sampler, num_workers=num_workers),
        'test': DataLoader(dataset_test,  batch_size=batch_size_segments, shuffle=False, num_workers=num_workers),
        }

# #     # Check if normalization worked
#     dataiter = iter(dataloaders['test'])
#     images_tmp, labels_tmp = dataiter.next()

#     print('Dim of sample batch sample: {}\n'.format(images_tmp.shape))
#     print('Dim of labels: {}\n'.format(labels_tmp.shape))
#     #print('max {}, min {}'.format(torch.max(images_tmp),torch.min(images_tmp)))
#     #print('max_labels {}, min {}'.format(torch.max(labels_tmp),torch.min(labels_tmp)))

#     fig, ax = plt.subplots(1,5,figsize=(20,5),facecolor='w', edgecolor='k')
#     ax =ax.ravel()
#     for idx in range(5):
#         ax[idx].plot(labels_tmp[idx,:].squeeze())
#     plt.show()
# #     plt.imshow(images_tmp[1].reshape((64,64)))
# #     plt.show()
    
    return dataloaders