import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import continuous_transformer.ContSpaceTime as ContSpaceTime
from continuous_transformer.continuous_utils import patch_sampling, plot_predictions, plot_training_curves, plot_whole_frame
from continuous_transformer.spectral_normalization import SpectralNorm
# import torch.nn.utils.spectral_norm as SpectralNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BERT_Model(pl.LightningModule):
    def __init__(self,
                 cfg, 
                 patch_sampling_cfg,
                 train_cfg,
                 n_labels=1, **kwargs):
        super().__init__()
        
        # -- sanity check: sampling parameter exists
        sampling_type = patch_sampling_cfg["structure"]

        if sampling_type not in ["grid", "random"]:
            print(f"sampling type '{sampling_type}' not recognized, using random instead")
            patch_sampling_cfg["structure"] = "random"

        # -- sanity check: model type exists
        model_type = cfg["model_type"]
        print('model_type: ',model_type)
        if model_type not in ["linear_encoder", "conv_encoder"]:
            print(f"model type '{sampling_type}' not recognized, using 'linea_encoder' instead")
            cfg["model_type"] = model_type = "linear_encoder"

        if model_type == "linear_encoder":
            self.model = BERT_LinearEncoder(cfg, n_labels)
#             print('cfg: ',cfg)
#             print('patch_sampling_cfg: ',patch_sampling_cfg)
        elif model_type == "conv_encoder":
            self.model = BERT_ConvEncoder(cfg, n_labels)
            
        # -- set plotting variable
        if cfg["plot_predictions"] == False:
            cfg["plotting"] = False
            
        self.Lipschitz_regularization = cfg["Lipschitz_regularization"]
        # -- save hyperparameters so they are accessible everywhere
        # -- access using self.hparams
        self.save_hyperparameters()

    def forward(self, latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos):
#         print('latentVar [models_lighting-forward].shape: ',latentVar.shape)
        verbose = False

        h, scores, embedded_patches = self.model.transformer(latentVar, T, P_row, P_col, map_fn, input_mask.float())
        
        embedded_patches_lastLayer = h.clone().detach()
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        masked_pos = masked_pos.type_as(latentVar)
        masked_pos = torch.as_tensor(masked_pos, dtype=torch.int64)
        
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.model.norm(self.model.activ2(self.model.linear(h_masked)))
        if self.Lipschitz_regularization:
            
#             # if self.current_epoch == 0: # If this is the first time 
#             if verbose: 
#                 print("h_masked.shape: ",h_masked.shape)
#                 print("self.model.beta_Lipschitz: ",self.model.beta_Lipschitz)
#                 print("self.model.decoder.weight.shape: ",self.model.decoder.weight.shape)
#                 print("self.model.encoder.weight.shape: ",self.model.encoder.weight.shape)
#                 print("self.model.decoder.weight.T: ",self.model.decoder.weight.T)
#                 print("self.model.gamma_Lipschitz: ", self.model.gamma_Lipschitz)
#                 print("self.model.I.shape: ",self.model.I.shape)
                
#             U_dec, S_dec, Vh_dec = torch.linalg.svd(self.model.decoder.weight)
#             S_dec = torch.clamp(S_dec, min=-1, max=1)
#             U_dec = self.model.beta_Lipschitz * (U_dec - U_dec.T) + (
#                 1 - self.model.beta_Lipschitz) * (U_dec +U_dec.T) - self.model.gamma_Lipschitz * torch.eye(U_dec.shape[0]).to(device)
#             Vh_dec = self.model.beta_Lipschitz * (Vh_dec - Vh_dec.T) + (
#                 1 - self.model.beta_Lipschitz) * (Vh_dec +Vh_dec.T) - self.model.gamma_Lipschitz * torch.eye(Vh_dec.shape[0]).to(device)
#             W1 = U_dec @ torch.diag(S_dec) @ Vh_dec[:U_dec.shape[0],:]
#             if verbose: 
#                 print("U_dec.shape: ",U_dec.shape)
#                 print("S_dec.shape: ",S_dec.shape)
#                 print("S_dec: ",S_dec)
#                 print("Vh_dec.shape: ",Vh_dec.shape)
#                 print('Vh_dec[:, :U_dec.shape[0]].shape: ',Vh_dec[:U_dec.shape[0],:].shape)
#                 print('torch.dist(self.model.decoder.weight, W1): ',torch.dist(self.model.decoder.weight, W1))
#                 print("W1.shape: ",W1.shape)
            
            # self.model.S_dec = torch.clamp(self.model.S_dec, min=-1, max=1)
        
            # #### Using SVD #####
            # self.model.S_dec.data.clamp_(min=-1, max=1)
            # self.model.U_dec.data = self.model.beta_Lipschitz * (self.model.U_dec - self.model.U_dec.T) + (
            #     1 - self.model.beta_Lipschitz) * (self.model.U_dec.data +self.model.U_dec.data.T) - self.model.gamma_Lipschitz * torch.eye(self.model.U_dec.shape[0]).to(device)
            # self.model.Vh_dec.data = self.model.beta_Lipschitz * (self.model.Vh_dec - self.model.Vh_dec.T) + (
            #     1 - self.model.beta_Lipschitz) * (self.model.Vh_dec.data +self.model.Vh_dec.T) - self.model.gamma_Lipschitz * torch.eye(self.model.Vh_dec.shape[0]).to(device)
            # self.model.W_dec.data = self.model.U_dec @ torch.diag(self.model.S_dec) @ self.model.Vh_dec[:self.model.U_dec.shape[0],:]
            # if verbose: print('self.model.W_dec: ',self.model.W_dec)
            # logits_lm=torch.matmul(h_masked, self.model.W_dec.T)
            # #### Using SVD #####
            
            param = self.model.decoder.weight
            sym = torch.mm(param, torch.t(param))
            sym -= torch.eye(param.shape[0]).to(device)
            ls_ort_dec = sym.pow(2.0).sum()  # Loss for orthogonality
            
            logits_lm = self.model.decoder(h_masked)
            if verbose: print('latentVar [models_lighting after Lipschitz].shape: ',latentVar.shape)
        else: 
            logits_lm = self.model.decoder(h_masked) #+ self.decoder_bias
            
        logits_lm = logits_lm.reshape(logits_lm.shape[0], logits_lm.shape[1], self.hparams.cfg["patch_size"], self.hparams.cfg["patch_size"])
#         logits_lm = logits_lm.squeeze()
        if self.Lipschitz_regularization:
            return logits_lm,scores, embedded_patches, embedded_patches_lastLayer, ls_ort_dec, ls_ort_pos_embed_3DLin, ls_ort_proj_q, ls_ort_proj_k, ls_ort_proj_v, ls_ort_proj, ls_ort_fc1, ls_ort_fc2
        else:
            return logits_lm,scores, embedded_patches, embedded_patches_lastLayer

    def custom_histogram_adder(self):
        # iterating through all parameters
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def training_step(self, batch, batch_idx):
        verbose = False
        # -- load variables from cfg dict
        n_pred = self.hparams.patch_sampling_cfg["num_patches_to_hide"]
        num_patches = self.hparams.patch_sampling_cfg["num_patches"]
        num_frames = self.hparams.patch_sampling_cfg["num_frames"]
        frame_size = self.hparams.cfg["frame_size"]
        patch_size = self.hparams.cfg["patch_size"]
        model_type = self.hparams.cfg["model_type"]
        scale_gauss = self.hparams.cfg["scale_gauss"]
        Lipschitz_regularization = self.hparams.cfg["Lipschitz_regularization"]
        penalty_orthogonality = self.hparams.cfg["penalty_orthogonality"]
        
        # -- batch has data on [0] and targets on [1]
        data = batch[0]
        if verbose: print('data [models_lighting].shape: ',data.shape)
#         data = torch.reshape(data, (data.shape[0], num_frames, frame_size, frame_size))
#         print('data [models_lighting].shape: ',data.shape)
        
        # -- sample patch in time and space
        # -- returns patch coordinates, segment with patches, and masked positions
        T, \
        P_row, \
        P_col, \
        segm_frames, \
        masked_pos, \
        masked_tokens = patch_sampling(data, self.hparams.patch_sampling_cfg, frame_size, patch_size, self.current_epoch)
#         masked_tokens = patch_sampling(data, self.hparams, frame_size, patch_size, self.current_epoch)
#         print('masked_tokens [models_lighting].shape: ',masked_tokens.shape)
        
        if model_type=='conv_encoder':
            segm_frames = segm_frames.reshape(segm_frames.shape[0],1,patch_size,patch_size)

        latentVar = self.model.encoder(segm_frames.type_as(data))
        
        
        if verbose: print('self.model.encoder [models_lighting without Lipschitz]: ',self.model.encoder.weight)
        if verbose: print('latentVar [models_lighting].shape: ',latentVar.shape)

        input_mask = torch.ones(latentVar.shape[0], latentVar.shape[1])
       
        mapping_size = 256
        if scale_gauss is None: 
            map_fn = None 
        else: 
            map_fn = np.random.normal(0, scale=1, size=(mapping_size, 3)) * scale_gauss
            
        
        logits_lm , _ , _, _ = self.forward(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
        
        if verbose:
            print('logits_lm.shape: ',logits_lm.shape)
            print('masked_tokens.shape: ', masked_tokens.shape)
        
        # logits_lm and masked_tokens should have the shape (batch, num_patches_hidden, patch)
        if logits_lm.dim()<4:
            logits_lm = logits_lm[None,:]
        if masked_tokens.dim()<4: 
            masked_tokens = masked_tokens[None,:]

        if verbose:
            print('logits_lm.shape: ',logits_lm.shape)
            print('masked_tokens.shape: ', masked_tokens.shape)
            
        logits_lm = logits_lm.reshape(logits_lm.shape[0],logits_lm.shape[1],patch_size,patch_size)
        masked_tokens = masked_tokens.reshape(masked_tokens.shape[0],masked_tokens.shape[1],patch_size,patch_size)
        
        loss_lm = F.mse_loss(logits_lm, masked_tokens) # for masked LM
#         print('logits_lm.shape: ',logits_lm.shape)
#         print('masked_tokens.shape: ', masked_tokens.shape)

        if logits_lm.dim()==1:
            logits_lm = logits_lm[None,:]
            masked_tokens = masked_tokens[None,:]

        tmp10 = []
        for idx_tmp in range(logits_lm.shape[0]):
            # Check if they are all the same value. If yes, add a small jitter just to be able to compute R2
            A=logits_lm[idx_tmp,:].detach().cpu().numpy().flatten()
            B=masked_tokens[idx_tmp,:].cpu().numpy().flatten()
#             if len(np.unique(A)):
#                 A=A+np.random.rand(len(A))*10e-10
#             if len(np.unique(B)):
#                 B=B+np.random.rand(len(B))*10e-10
            r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2
        
            tmp10.append(r2_sq)
        mean_r2 = np.nanmean(tmp10)
        
        if Lipschitz_regularization:
            loss = loss_lm.float() + penalty_orthogonality*(ls_ort_dec.float() + ls_ort_enc.float() + ls_ort_pos_embed_3DLin.float() + ls_ort_proj_q.float() + 
                                                            ls_ort_proj_k.float() + ls_ort_proj_v.float() + ls_ort_proj.float() + ls_ort_fc1.float() + ls_ort_fc2.float())
        else:
            loss = loss_lm.float()
        loss = loss.type_as(data)
        self.log("train_loss", loss)
        self.log("train_r2", mean_r2)
        
        #Logs
        logs={"train_loss": loss, "train_r2": mean_r2}
        
        if Lipschitz_regularization:
            batch_dictionary={
                #REQUIRED: It ie required for us to return "loss"
                "loss": loss,
                "train_ls_ort_dec" : ls_ort_dec.float(),
                "train_ls_ort_enc" : ls_ort_enc.float(),
                "train_ls_ort_pos_embed_3DLin" : ls_ort_pos_embed_3DLin.float(),
                "train_ls_ort_proj_q" : ls_ort_proj_q.float(),
                "train_ls_ort_proj_k" : ls_ort_proj_k.float(),
                "train_ls_ort_proj_v" : ls_ort_proj_v.float(),
                "train_ls_ort_proj": ls_ort_proj.float(),
                "train_ls_ort_fc1" : ls_ort_fc1.float(),
                "train_ls_ort_fc2" : ls_ort_fc2.float(),
                "train_loss_lm" : loss_lm.float(),
                "train_r2": mean_r2,
                #optional for logging purposes
                "log": logs
              }
        else: 
            batch_dictionary={
                #REQUIRED: It ie required for us to return "loss"
                "loss": loss,
                "train_r2": mean_r2,
                #optional for logging purposes
                "log": logs
              }


        return batch_dictionary  #It was 'loss', now it is a dict

    def on_validation_epoch_start(self):
        # -- set ploting variables
        if self.hparams.cfg["plot_predictions"] == True:
            self.hparams.cfg["plotting"] = True
        if self.hparams.cfg["plot_training_curves"] == True:
            self.hparams.cfg["plot_training_curves"] = True

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # -- only plot for one batch, then turn off plotting
        self.hparams.cfg["plotting"] = False
        self.hparams.cfg["plot_training_curves"] = False
        
    def validation_epoch_end(self, outputs):
        Lipschitz_regularization = self.hparams.cfg["Lipschitz_regularization"]
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if Lipschitz_regularization:
            avg_lm_loss = torch.stack([x['val_loss_lm'] for x in outputs]).mean()
            avg_ls_ort_dec = torch.stack([x['val_ls_ort_dec'] for x in outputs]).mean()
            avg_ls_ort_enc = torch.stack([x['val_ls_ort_enc'] for x in outputs]).mean()
            avg_ls_ort_pos_embed_3DLin = torch.stack([x['val_ls_ort_pos_embed_3DLin'] for x in outputs]).mean()
            avg_ls_ort_proj_q = torch.stack([x['val_ls_ort_proj_q'] for x in outputs]).mean()
            avg_ls_ort_proj_k = torch.stack([x['val_ls_ort_proj_k'] for x in outputs]).mean()
            avg_ls_ort_proj_v = torch.stack([x['val_ls_ort_proj_v'] for x in outputs]).mean()
            avg_ls_ort_proj = torch.stack([x['val_ls_ort_proj'] for x in outputs]).mean()
            avg_ls_ort_fc1 = torch.stack([x['val_ls_ort_fc1'] for x in outputs]).mean()
            avg_ls_ort_fc2 = torch.stack([x['val_ls_ort_fc2'] for x in outputs]).mean()
        

#         val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        if Lipschitz_regularization:
            tensorboard_logs = {'val_loss': avg_loss, 
                                'val_loss_lm': avg_lm_loss,
                                'val_ls_ort_dec': avg_ls_ort_dec,
                                'val_ls_ort_enc': avg_ls_ort_enc,
                                "val_ls_ort_pos_embed_3DLin" : avg_ls_ort_pos_embed_3DLin.float(),
                                "val_ls_ort_proj_q" : avg_ls_ort_proj_q.float(),
                                "val_ls_ort_proj_k" : avg_ls_ort_proj_k.float(),
                                "val_ls_ort_proj_v" : avg_ls_ort_proj_v.float(),
                                "val_ls_ort_proj": avg_ls_ort_proj.float(),
                                "val_ls_ort_fc1" : avg_ls_ort_fc1.float(),
                                "val_ls_ort_fc2" : avg_ls_ort_fc2.float(),
                                'step': self.current_epoch,
                               }
        else: 
            tensorboard_logs = {'val_loss': avg_loss, 
                                'step': self.current_epoch,
                               }
    
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
        if Lipschitz_regularization:
            self.logger.experiment.add_scalar("Loss/Val_LM", avg_lm_loss, self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Val_ort_dec", avg_ls_ort_dec, self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Val_ort_enc", avg_ls_ort_enc, self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Val_ort_pos_embed_3DLin",avg_ls_ort_pos_embed_3DLin,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Val_ort_proj_q",avg_ls_ort_proj_q,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Val_ort_proj_k",avg_ls_ort_proj_k,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Val_ort_proj_v",avg_ls_ort_proj_v,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Val_ort_proj",avg_ls_ort_proj,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Val_ort_fc1",avg_ls_ort_fc1,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Val_ort_fc2",avg_ls_ort_fc2,self.current_epoch)
        

        return {'log': tensorboard_logs}
    
    def training_epoch_end(self, outputs):
        Lipschitz_regularization = self.hparams.cfg["Lipschitz_regularization"]
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        if Lipschitz_regularization:
            avg_lm_loss = torch.stack([x['train_loss_lm'] for x in outputs]).mean()
            avg_ls_ort_dec = torch.stack([x['train_ls_ort_dec'] for x in outputs]).mean()
            avg_ls_ort_enc = torch.stack([x['train_ls_ort_enc'] for x in outputs]).mean()
            avg_ls_ort_pos_embed_3DLin = torch.stack([x['train_ls_ort_pos_embed_3DLin'] for x in outputs]).mean()
            avg_ls_ort_proj_q = torch.stack([x['train_ls_ort_proj_q'] for x in outputs]).mean()
            avg_ls_ort_proj_k = torch.stack([x['train_ls_ort_proj_k'] for x in outputs]).mean()
            avg_ls_ort_proj_v = torch.stack([x['train_ls_ort_proj_v'] for x in outputs]).mean()
            avg_ls_ort_proj = torch.stack([x['train_ls_ort_proj'] for x in outputs]).mean()
            avg_ls_ort_fc1 = torch.stack([x['train_ls_ort_fc1'] for x in outputs]).mean()
            avg_ls_ort_fc2 = torch.stack([x['train_ls_ort_fc2'] for x in outputs]).mean()

        if Lipschitz_regularization:
            tensorboard_logs = {'train_loss': avg_loss, 
                                'train_loss_lm': avg_lm_loss,
                                'train_ls_ort_dec': avg_ls_ort_dec,
                                'train_ls_ort_enc': avg_ls_ort_enc,
                                "train_ls_ort_pos_embed_3DLin" : avg_ls_ort_pos_embed_3DLin.float(),
                                "train_ls_ort_proj_q" : avg_ls_ort_proj_q.float(),
                                "train_ls_ort_proj_k" : avg_ls_ort_proj_k.float(),
                                "train_ls_ort_proj_v" : avg_ls_ort_proj_v.float(),
                                "train_ls_ort_proj": avg_ls_ort_proj.float(),
                                "train_ls_ort_fc1" : avg_ls_ort_fc1.float(),
                                "train_ls_ort_fc2" : avg_ls_ort_fc2.float(),
                                'step': self.current_epoch,
                               }
        else: 
            tensorboard_logs = {'train_loss': avg_loss, 
                                'step': self.current_epoch,
                               }
        # logging histograms
        self.custom_histogram_adder()
    
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",avg_loss,self.current_epoch)
        if Lipschitz_regularization:
            self.logger.experiment.add_scalar("Loss/Train_LM",avg_lm_loss,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_dec",avg_ls_ort_dec,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_enc",avg_ls_ort_enc,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_pos_embed_3DLin",avg_ls_ort_pos_embed_3DLin,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_proj_q",avg_ls_ort_proj_q,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_proj_k",avg_ls_ort_proj_k,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_proj_v",avg_ls_ort_proj_v,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_proj",avg_ls_ort_proj,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_fc1",avg_ls_ort_fc1,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_fc2",avg_ls_ort_fc2,self.current_epoch)
        
        
    
        epoch_dictionary={
            # required
            'loss': avg_loss,
            # for logging purposes
            'log': tensorboard_logs}

        
    def validation_step(self, batch, batch_idx):
        verbose = False
        # -- load variables from cfg dict
        n_pred = self.hparams.patch_sampling_cfg["num_patches_to_hide"]
        num_patches = self.hparams.patch_sampling_cfg["num_patches"]
        num_frames = self.hparams.patch_sampling_cfg["num_frames"]
        frame_size = self.hparams.cfg["frame_size"]
        patch_size = self.hparams.cfg["patch_size"]
        model_type = self.hparams.cfg["model_type"]
        scale_gauss = self.hparams.cfg["scale_gauss"]
        Lipschitz_regularization = self.hparams.cfg["Lipschitz_regularization"]
        penalty_orthogonality = self.hparams.cfg["penalty_orthogonality"]
        
        # -- batch has data on [0] and targets on [1]
        data = batch[0]
        if verbose: print('data [models_lighting].shape: ',data.shape)
#         data = torch.reshape(data, (data.shape[0], num_frames, frame_size["rows"], frame_size["cols"]))
#         if verbose: print('data [models_lighting].shape: ',data.shape)
        
        # -- sample patch in time and space
        # -- returns patch coordinates, segment with patches, and masked positions
        T, \
        P_row, \
        P_col, \
        segm_frames, \
        masked_pos, \
        masked_tokens = patch_sampling(data, self.hparams.patch_sampling_cfg, frame_size, patch_size)
#         print('[models_light] T.shape: {}, P_row.shape: {}, P_col.shape: {}, segm_frames.shape: {}, masked_pos.shape: {}, masked_tokens.shape: {}'.format(T.shape, P_row.shape, P_col.shape, segm_frames.shape, masked_pos.shape, masked_tokens.shape))
        
        if model_type=='conv_encoder':
            segm_frames = segm_frames.reshape(segm_frames.shape[0],1,patch_size,patch_size)

        if verbose: print('segm_frames.shape: ',segm_frames.shape)
        
        if Lipschitz_regularization:
            # U_enc, S_enc, Vh_enc = torch.linalg.svd(self.model.encoder.weight)
            # S_enc = torch.clamp(S_enc, min=-1, max=1)
            # U_enc = self.model.beta_Lipschitz * (U_enc - U_enc.T) + (
            #     1 - self.model.beta_Lipschitz) * (U_enc +U_enc.T) - self.model.gamma_Lipschitz * torch.eye(U_enc.shape[0]).to(device)
            # Vh_enc = self.model.beta_Lipschitz * (Vh_enc - Vh_enc.T) + (
            #     1 - self.model.beta_Lipschitz) * (Vh_enc +Vh_enc.T) - self.model.gamma_Lipschitz * torch.eye(Vh_enc.shape[0]).to(device)
            # W2 = U_enc @ torch.diag(S_enc) @ Vh_enc[:U_enc.shape[0],:]
            # if verbose: 
            #     print("self.model.U_enc.shape: ",self.model.U_enc.shape)
            #     print("self.model.S_enc.shape: ",self.model.S_enc.shape)
            #     print("self.model.S_enc: ",self.model.S_enc)
                # print("Vh_enc.shape: ",Vh_enc.shape)
                # print('Vh_enc[:, :U_enc.shape[0]].shape: ',Vh_enc[:U_enc.shape[0],:].shape)
                # print('torch.dist(self.model.decoder.weight, W2): ',torch.dist(self.model.encoder.weight, W2))
                # print("W2.shape: ",W2.shape)
            
            # ## To use the SVD directly
            # # self.model.S_enc = torch.clamp(self.model.S_enc, min=-1, max=1)
            # # temp = torch.clamp(self.model.S_enc, min=-1, max=1)
            # # print("temp: ",temp)
            # # self.model.S_enc = nn.Parameter(torch.clamp(self.model.S_enc, min=-1, max=1))
            # self.model.S_enc.data.clamp_(min=-1, max=1)
            # print('self.model.S_enc: ',self.model.S_enc)
            # W2 = self.model.U_enc @ torch.diag(self.model.S_enc) @ self.model.Vh_enc[:self.model.U_enc.shape[0],:]

            # # logits_lm=torch.matmul(h_masked, W2.T)
            # latentVar=torch.matmul(segm_frames.type_as(data), W2.T)
            # print('latentVar: ',latentVar)
            param = self.model.encoder.weight
            sym = torch.mm(param, torch.t(param))
            sym -= torch.eye(param.shape[0]).to(device)
            ls_ort_enc = sym.pow(2.0).sum()  # Loss for orthogonality
            latentVar = self.model.encoder(segm_frames.type_as(data))
                                   
        else:      
            latentVar = self.model.encoder(segm_frames.type_as(data))
#         latentVar = latentVar.reshape(1, latentVar.shape[0], latentVar.shape[1])
            if verbose: print('latentVar [models_lighting].shape: ',latentVar.shape)

        # latentVar = self.model.encoder(segm_frames.type_as(data))
        if verbose: print('latentVar.shape: ',latentVar.shape)
#         latentVar = latentVar.reshape(1, latentVar.shape[0], latentVar.shape[1])
#         print('latentVar [models_lighting - validation step].shape: ',latentVar.shape)

#         # original 
#         input_mask = torch.zeros(1, latentVar.shape[1])
#         input_mask[:, :segm_frames.shape[0]] = 1

#         x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
#         mask : (B(batch_size) x S(seq_len))
        input_mask = torch.ones(latentVar.shape[0], latentVar.shape[1])
        if verbose: print('input_mask.shape: ',input_mask.shape)
#         input_mask[:, :segm_frames.shape[0]] = 1
       
        mapping_size = 256
#         scale_gauss = 1
#         print('scale_gauss: ', scale_gauss)
        if scale_gauss is None: 
            map_fn = None 
        else: 
            map_fn = np.random.normal(0, scale=1, size=(mapping_size, 3)) * scale_gauss
            

        if Lipschitz_regularization:
            logits_lm , _ , _, _, ls_ort_dec, ls_ort_pos_embed_3DLin, ls_ort_proj_q, ls_ort_proj_k, ls_ort_proj_v, ls_ort_proj, ls_ort_fc1, ls_ort_fc2 = self.forward(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
        else:
            logits_lm , _ , _, _ = self.forward(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
            
        if verbose:
            print('logits_lm.shape: ',logits_lm.shape)
            print('masked_tokens.shape: ', masked_tokens.shape)
        
        # logits_lm and masked_tokens should have the shape (batch, num_patches_hidden, patch)
        if logits_lm.dim()<4:
            logits_lm = logits_lm[None,:]
        if masked_tokens.dim()<4: 
            masked_tokens = masked_tokens[None,:]
            
        logits_lm = logits_lm.reshape(logits_lm.shape[0],logits_lm.shape[1],patch_size,patch_size)
        masked_tokens = masked_tokens.reshape(masked_tokens.shape[0],masked_tokens.shape[1],patch_size,patch_size)
            
        if verbose:
            print('logits_lm.shape: ',logits_lm.shape)
            print('masked_tokens.shape: ', masked_tokens.shape)
        loss_lm = F.mse_loss(logits_lm, masked_tokens) # for masked LM

        if logits_lm.dim()==1:
            logits_lm = logits_lm[None,:]
            masked_tokens = masked_tokens[None,:]

        tmp10 = []
        for idx_tmp in range(logits_lm.shape[0]):
#             tmp10.append((np.corrcoef(logits_lm[idx_tmp,:].detach().cpu().numpy().flatten(), masked_tokens[idx_tmp,:].cpu().numpy().flatten(), rowvar=False)[0,1])**2)
             # Check if they are all the same value. If yes, add a small jitter just to be able to compute R2
            A=logits_lm[idx_tmp,:].detach().cpu().numpy().flatten()
            B=masked_tokens[idx_tmp,:].cpu().numpy().flatten()
#             if len(np.unique(A)):
#                 A=A+np.random.rand(len(A))*10e-10
#             if len(np.unique(B)):
#                 B=B+np.random.rand(len(B))*10e-10
            r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2
        
            tmp10.append(r2_sq)
        mean_r2 = np.nanmean(tmp10)

#         print('tmp10: ',tmp10)
#         print('mean_r2; ',mean_r2)
        if Lipschitz_regularization:
            loss = loss_lm.float() + penalty_orthogonality*(ls_ort_dec.float() + ls_ort_enc.float() + ls_ort_pos_embed_3DLin.float() + ls_ort_proj_q.float() + 
                                                            ls_ort_proj_k.float() + ls_ort_proj_v.float() + ls_ort_proj.float() + ls_ort_fc1.float() + ls_ort_fc2.float())
        else:
            loss = loss_lm.float()
            
        loss = loss.type_as(data)
        self.log("val_loss", loss)
        self.log("val_r2", mean_r2)
        
        #Logs
        logs={"val_loss": loss, "val_r2": mean_r2}
        
        if Lipschitz_regularization:
            batch_dictionary={
                #REQUIRED: It ie required for us to return "loss"
                "val_loss": loss,
                "val_ls_ort_dec" : ls_ort_dec.float(),
                "val_ls_ort_enc" : ls_ort_enc.float(),
                "val_ls_ort_pos_embed_3DLin" : ls_ort_pos_embed_3DLin.float(),
                "val_ls_ort_proj_q" : ls_ort_proj_q.float(),
                "val_ls_ort_proj_k" : ls_ort_proj_k.float(),
                "val_ls_ort_proj_v" : ls_ort_proj_v.float(),
                "val_ls_ort_proj": ls_ort_proj.float(),
                "val_ls_ort_fc1" : ls_ort_fc1.float(),
                "val_ls_ort_fc2" : ls_ort_fc2.float(),
                "val_loss_lm" : loss_lm.float(),
                "val_r2": mean_r2,
                #optional for logging purposes
                "log": logs
              }
        else: 
            batch_dictionary={
                #REQUIRED: It ie required for us to return "loss"
                "val_loss": loss,
                "val_r2": mean_r2,
                #optional for logging purposes
                "log": logs
              }
        
        plotting = self.hparams.cfg["plotting"]
        epochs_plot = self.hparams.cfg["plot_every_epochs"]
                    
        
        if plotting == True and self.current_epoch%epochs_plot == 0:
            plot_predictions(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,
                             self.current_epoch, loss, mean_r2, self.logger.log_dir)
            plot_whole_frame(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size, frame_size,
                             self.current_epoch, loss, mean_r2, self.logger.log_dir)
            
        if self.hparams.cfg["plot_training_curves"] == True: 
            plot_training_curves(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size, 
                     self.current_epoch, loss, mean_r2, self.logger.log_dir)
            
        return batch_dictionary  #It was 'loss', now it is a dict
    

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.hparams.train_cfg["learning_rate"],
                         weight_decay=self.hparams.train_cfg["weight_decay"])
        return optimizer
    
    
    
'''
Experiment with BERT model from this github:
https://github.com/dhlee347/pytorchic-bert
'''
class CNN_arc(nn.Module):     
    def __init__(self, image_size):
        super(CNN_arc, self).__init__()
        self.conv0 = nn.Conv2d(1, 32, 4, stride=2, padding=1)  # [(image_size−K+2P)/S]+1 = [(image_size−4+2*1)/2]+1
        self.batch0 =  nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  #[([[(image_size−4+2*1)/2]+1]−4+2*1)/2]+1
        self.batch1 =  nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 
        self.batch2 = nn.BatchNorm2d(128)
#         in_features = int((((((((((((image_size-4+2*1)/2)+1)-4+2*1)/2)+1)-4+2*1)/2)+1)-4+2*1)/2)+1)
        in_features = int(((image_size/8)**2)*128)
#         print('image_size: ',image_size)
#         print('in_features: ',in_features)
        self.fc1 = nn.Linear(in_features=in_features, out_features=768)   #For the 16x16 patch
        self.batch3 = nn.BatchNorm1d(768)

        nn.init.kaiming_normal_(self.conv0.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv0.bias, 0)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.batch0.weight, 1)
        nn.init.constant_(self.batch0.bias, 0)
        nn.init.constant_(self.batch1.weight, 1)
        nn.init.constant_(self.batch1.bias, 0)
        nn.init.constant_(self.batch2.weight, 1)
        nn.init.constant_(self.batch2.bias, 0)
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.batch3.weight, 1)
        nn.init.constant_(self.batch3.bias, 0)
            
    def forward(self, images):
#         print('\nimages.shape: ',images.shape)
        x = F.leaky_relu(self.conv0(images)) # [64,64,1], K=4, P=1, S=2 -> W2 =(W−K+2P)/S+1 = (64-3+2*1)/2+1= 86
#         print('x.shape: ',x.shape)
        x = self.batch0(x)
        x = F.leaky_relu(self.conv1(x)) # [256,256,1], K=3, P=1, S=3 -> W2 =(W−K+2P)/S+1 = (256-3+2*1)/3+1= 86
#         print('x1.shape: ',x.shape)
        x = self.batch1(x)
        x = F.leaky_relu(self.conv2(x)) #[43,43,16] -> W2 = (43-3+2*1)/2+1 = 22
#         print('x2.shape: ',x.shape)
        x = self.batch2(x)
        x = x.view([images.size(0), -1])
#         print('x3.shape: ',x.shape)
        x = F.leaky_relu(self.fc1(x)) #size mismatch, m1: [128 x 200], m2: [3528 x 500] at 
        x = self.batch3(x)
        return x #, p1,self_att

class BERT_LinearEncoder(nn.Module):
    "Bert Model : Masked LM and next sentence classification"
    def __init__(self, cfg, n_labels=1):
        super(BERT_LinearEncoder, self).__init__()
        self.transformer = ContSpaceTime.Transformer(cfg)
#         self.fc = nn.Linear(cfg["dim"], cfg["dim"])
        self.activ1 = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(cfg["p_drop_hidden"])
        self.activ2 = ContSpaceTime.gelu
        self.norm = ContSpaceTime.LayerNorm(cfg)
        self.beta_Lipschitz=0.8
        self.gamma_Lipschitz=0.01
        # self.I = torch.eye(cfg["patch_size"]**2) #For the encoder
#         n_dim = cfg["dim"]
        dim_images = cfg["patch_size"]**2
    
        if cfg["Lipschitz_regularization"]:
            self.linear = nn.Linear(cfg["dim"], cfg["dim"])
            self.decoder = nn.Linear(cfg["dim"], dim_images, bias=False)
            if cfg["operation_with_pos_encoding"] == "concatenate":
                self.encoder = nn.Linear(dim_images, int(cfg["dim"]/2), bias=False)
            elif cfg["operation_with_pos_encoding"] == "sum":  
                self.encoder = nn.Linear(dim_images, cfg["dim"], bias=False)
            nn.init.orthogonal_(self.decoder.weight)
            nn.init.orthogonal_(self.encoder.weight)
            nn.init.orthogonal_(self.linear.weight)
            
#               #### Using SVD #####
# #             self.decoder = nn.Linear(cfg["dim"], dim_images, bias=False)
# #             U_dec, S_dec, Vh_dec = torch.linalg.svd(self.decoder.weight) #Do this just to get the right dimensions for USV. 
            
# #             self.U_dec = nn.Parameter(U_dec)
# #             self.S_dec = nn.Parameter(S_dec)
# #             nn.init.normal_(self.S_dec)
# #             self.Vh_dec = nn.Parameter(Vh_dec)
# #             del self.decoder
# #             self.W_dec = self.U_dec @ torch.diag(self.S_dec) @ self.Vh_dec[:self.U_dec.shape[0],:]
            
# #             if cfg["operation_with_pos_encoding"] == "concatenate":
# #                 self.encoder = nn.Linear(dim_images, int(cfg["dim"]/2), bias=False)
# #             elif cfg["operation_with_pos_encoding"] == "sum":  
# #                 self.encoder = nn.Linear(dim_images, cfg["dim"], bias=False)
                
# #             U_enc, S_enc, Vh_enc = torch.linalg.svd(self.encoder.weight) #Do this just to get the right dimensions for USV. 
# #             self.U_enc = nn.Parameter(U_enc)
# #             self.S_enc = nn.Parameter(S_enc)
# #             nn.init.normal_(self.S_enc)
# #             self.Vh_enc = nn.Parameter(Vh_enc)
# #             del self.encoder
# #             self.W_enc = self.U_enc @ torch.diag(self.S_enc) @ self.Vh_enc[:self.U_enc.shape[0],:]
# #             #### \Using SVD #####
            
        elif cfg["Spectral_normalization"]: #To use spectral normalization
            self.linear = SpectralNorm(nn.Linear(cfg["dim"], cfg["dim"]))
            self.decoder = SpectralNorm(nn.Linear(cfg["dim"], dim_images, bias=False))
            if cfg["operation_with_pos_encoding"] == "concatenate":
                self.encoder = SpectralNorm(nn.Linear(dim_images, int(cfg["dim"]/2), bias=False))
            elif cfg["operation_with_pos_encoding"] == "sum":  
                self.encoder = SpectralNorm(nn.Linear(dim_images, cfg["dim"], bias=False))
            
            # self.linear = nn.utils.parametrizations.spectral_norm(nn.Linear(cfg["dim"], cfg["dim"]))
            # self.decoder = nn.utils.parametrizations.spectral_norm(nn.Linear(cfg["dim"], dim_images, bias=False))
            # if cfg["operation_with_pos_encoding"] == "concatenate":
            #     self.encoder = nn.utils.parametrizations.spectral_norm(nn.Linear(dim_images, int(cfg["dim"]/2), bias=False))
            # elif cfg["operation_with_pos_encoding"] == "sum":  
            #     self.encoder = nn.utils.parametrizations.spectral_norm(nn.Linear(dim_images, cfg["dim"], bias=False))
                
#             nn.init.orthogonal_(self.decoder.weight)
#             nn.init.orthogonal_(self.encoder.weight)
#             nn.init.orthogonal_(self.linear.weight)
            
        else: #If not Lipschitz constrained, just use linear layers
            self.linear = nn.Linear(cfg["dim"], cfg["dim"])
            self.decoder = nn.Linear(cfg["dim"], dim_images, bias=False)
            # print("cfg[""operation_with_pos_encoding""]: ", cfg["operation_with_pos_encoding"])
            if cfg["operation_with_pos_encoding"] == "concatenate":
                self.encoder = nn.Linear(dim_images, int(cfg["dim"]/2), bias=False)
            elif cfg["operation_with_pos_encoding"] == "sum":  
                self.encoder = nn.Linear(dim_images, cfg["dim"], bias=False)
            nn.init.xavier_normal_(self.decoder.weight, gain=nn.init.calculate_gain('tanh'))


class BERT_ConvEncoder(nn.Module):
    "Bert Model : Masked LM and next sentence classification"
    def __init__(self, cfg, n_labels=1):
        super().__init__()
#         print(f"BERT_ConvEncoder IS NOT IMPLEMENTED")
        self.transformer = ContSpaceTime.Transformer(cfg)
        self.fc = nn.Linear(cfg["dim"], cfg["dim"])
        self.activ1 = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(cfg["p_drop_hidden"])
        self.linear = nn.Linear(cfg["dim"], cfg["dim"])
        self.activ2 = ContSpaceTime.gelu
        self.norm = ContSpaceTime.LayerNorm(cfg)
        n_dim = cfg["dim"]
        dim_images = cfg["patch_size"]**2
        self.decoder = nn.Linear(n_dim, dim_images, bias=False)
#         self.encoder = nn.Linear(dim_images, n_dim, bias=False)
        self.encoder = CNN_arc(cfg["patch_size"])
#         nn.init.kaiming_normal_(self.decoder.weight, mode='fan_out', nonlinearity='relu')

