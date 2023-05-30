import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import continuous_transformer.ContSpaceTime as ContSpaceTime
from continuous_transformer.continuous_utils import patch_sampling, plot_predictions, plot_training_curves, plot_whole_frame, plot_whole_sequence, plot_predictions_curves, plot_attention_weights
from continuous_transformer.utils import SaveBestModel_CST

from continuous_transformer.sobolev_loss import sobolev_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BERT_Model(pl.LightningModule):
    def __init__(self,
                 cfg, 
                 patch_sampling_cfg,
                 train_cfg,my_loss=None,
                 path_to_save_models=None,
                 warmup=100,
                 n_labels=1, **kwargs):
        super().__init__()
        
        # -- sanity check: sampling parameter exists
        sampling_type = patch_sampling_cfg["structure"]
        self.in_between_frame_init = patch_sampling_cfg["in_between_frame_init"]

        self.sobolev_loss_ = my_loss
        
        self.save_best_model = SaveBestModel_CST()
        self.path_to_save_models = path_to_save_models
        self.val_loss = []
        self.warmup=warmup
        self.epoch_counter=0

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

        elif model_type == "conv_encoder":
            self.model = BERT_ConvEncoder(cfg, n_labels)
            
        # -- set plotting variable
        if cfg["plot_predictions"] == False:
            cfg["plotting"] = False
            
        # self.Lipschitz_regularization = False #cfg["Lipschitz_regularization"]
        # -- save hyperparameters so they are accessible everywhere
        # -- access using self.hparams
        self.save_hyperparameters()

    def forward(self, latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos):
        verbose=False
            
        h, scores, embedded_patches = self.model.transformer(latentVar, T, P_row, P_col, map_fn, input_mask.float())
        embedded_patches_lastLayer = h.clone().detach()
            
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        masked_pos = masked_pos.type_as(latentVar)
        masked_pos = torch.as_tensor(masked_pos, dtype=torch.int64)
        h_masked = torch.gather(h, 1, masked_pos) # Selects the tokens specified by 'masked_pos' from 'h'
        if verbose: 
            print('[in forward] masked_pos.shape: ',masked_pos.shape)
            print('[in forward] h_masked [normal].shape: ',h_masked.shape)
        h_masked = self.model.norm(self.model.activ2(self.model.linear(h_masked)))
            
        logits_lm = self.model.decoder(h_masked)
                
            
        logits_lm = logits_lm.reshape(logits_lm.shape[0], logits_lm.shape[1], self.hparams.cfg["patch_size"][0], self.hparams.cfg["patch_size"][1])
        return logits_lm,scores, embedded_patches, embedded_patches_lastLayer

    def custom_histogram_adder(self):
        # iterating through all parameters
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def training_step(self, batch, batch_idx):
        verbose=False
        # -- load variables from cfg dict
        n_pred = self.hparams.patch_sampling_cfg["num_patches_to_hide"]
        num_patches = self.hparams.patch_sampling_cfg["num_patches"]
        num_frames = self.hparams.patch_sampling_cfg["num_frames"]
        frame_size = self.hparams.cfg["frame_size"]
        patch_size = self.hparams.cfg["patch_size"]
        range_imshow = self.hparams.cfg["range_imshow"]
        model_type = self.hparams.cfg["model_type"]
        scale_gauss = self.hparams.cfg["scale_gauss"]
        penalty_orthogonality = self.hparams.cfg["penalty_orthogonality"]
        in_between_frame_init = self.hparams.patch_sampling_cfg["in_between_frame_init"]
        experiment_name = self.hparams.train_cfg["experiment_name"]
        compute_loss_whole_curve = self.hparams.train_cfg["compute_loss_whole_curve"]
        compute_loss_on_dummy_points = self.hparams.train_cfg["compute_loss_on_dummy_points"]
        if compute_loss_on_dummy_points:
            weight_loss_on_real = self.hparams.train_cfg["weight_loss_on_real"]
        std_noise_t = self.hparams.train_cfg["std_noise_t"]
        std_to_data = self.hparams.train_cfg["std_to_data"]
        plotting = self.hparams.cfg["plotting"]
        epochs_plot = self.hparams.cfg["plot_every_epochs"]
        
        # -- batch has data on [0] and targets on [1]
        data = batch[0]
        data.requires_grad=True 
        T_orig = batch[1][0].cpu() #Set T=None if you don't want to use original time coordinates
        
        # -- sample patch in time and space
        # -- returns patch coordinates, segment with patches, and masked positions
        T, \
        P_row, \
        P_col, \
        segm_frames, \
        masked_pos, \
        masked_tokens = patch_sampling(data, T_orig, self.hparams.patch_sampling_cfg, frame_size, patch_size, self.current_epoch)

        segm_frames = segm_frames.to(device).requires_grad_(True)
        if compute_loss_on_dummy_points: #Make copy of the segm_frames to compute the loss later
            segm_frames_orig = segm_frames.clone()

        if std_to_data>0:
            segm_frames_before_perturn = segm_frames.clone()
            perturb_0 = torch.normal(mean=torch.zeros_like(segm_frames),std=std_to_data)
            segm_frames = segm_frames + perturb_0
        
        if std_noise_t>0:
            T_before_perturn = T.clone()
            perturb = torch.normal(mean=torch.zeros_like(T),std=std_noise_t)
            T = T+perturb

        if plotting == True and self.current_epoch%epochs_plot == 0:
            dev_logits_lm=None
            logits_lm=None
            loss=None
            mean_r2=None
            if "Navier" in experiment_name:
                plot_predictions(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, segm_frames, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='train')
            else:
                plot_predictions_curves(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, None, logits_lm, segm_frames, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='train')
        
        if model_type=='conv_encoder':
            segm_frames = segm_frames.reshape(segm_frames.shape[0],1,patch_size,patch_size)
            
        latentVar = self.model.encoder(segm_frames.type_as(data))
        if verbose: print('[train] latentVar [models_lighting].shape: ',latentVar.shape)

        input_mask = torch.ones(latentVar.shape[0], latentVar.shape[1])
        if verbose: print('input_mask.shape: ',input_mask.shape)
       
        mapping_size = 256
        if scale_gauss is None: 
            map_fn = None 
        else: 
            map_fn = np.random.normal(0, scale=1, size=(mapping_size, 3)) * scale_gauss
        

        if verbose:
            print('[train] masked_tokens.shape: ', masked_tokens.shape)
    
        
        if self.sobolev_loss_ is not None:
            masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
            logits_lm_with_dummy , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
            
            if std_noise_t>0:
                    ids = np.argwhere(np.isin(T_before_perturn,T_orig,invert=False)).flatten()
            else:
                ids = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
            ids2 = np.argwhere(np.isin(T,T_orig,invert=True)).flatten()
            
            loss_lm = self.sobolev_loss_.evaluate__loss(y=logits_lm_with_dummy.squeeze(-1),
                                                        data=data.squeeze(-1),
                                                        x=(segm_frames),
                                                        x_fd=T.to(device),
                                                        y_0=logits_lm_with_dummy.squeeze(-1)[:,ids,:],
                                                        indexes = ids2
                                                        )
        else: #if it is not sobolev
            if compute_loss_whole_curve: 
                masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                scores2 = torch.stack(scores) 
                all_att = scores2.reshape(-1,scores2.shape[-1])

                if std_noise_t>0:
                    ids = np.argwhere(np.isin(T_before_perturn,T_orig,invert=False)).flatten()
                else: 
                    ids = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                
                loss_lm = F.mse_loss(logits_lm_with_dummy[:,ids,:], data) # for masked LM
            elif compute_loss_on_dummy_points:
                masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                if verbose: print('masked_pos_with_dummy.shape: ',masked_pos_with_dummy.shape)

                logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                scores2 = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
                all_att = scores2.reshape(-1,scores2.shape[-1])
                

                if verbose: 
                    print('T.shape: ',T.shape)
                    print('T: ',T)
                    print('logits_lm_with_dummy.shape: ', logits_lm_with_dummy.shape)
                    print('segm_frames.shape: ', segm_frames.shape)
                
                if std_noise_t>0:
                    ids_reals = np.argwhere(np.isin(T_before_perturn,T_orig,invert=False)).flatten()
                    ids_dummies = np.argwhere(np.isin(T_before_perturn,T_orig,invert=True)).flatten()
                else: 
                    ids_reals = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                    ids_dummies = np.argwhere(np.isin(T,T_orig,invert=True)).flatten()
                if verbose: 
                    print('ids_reals: ', ids_reals)
                    print('T[ids_reals]: ',T[ids_reals])
                    if std_noise_t>0: print('T_before_perturn[ids_reals]: ',T_before_perturn[ids_reals])
                    print('logits_lm_with_dummy[:,ids_reals,:]: ', logits_lm_with_dummy[:,ids_reals,:])
                loss_lm_real = F.mse_loss(logits_lm_with_dummy[:,ids_reals,:].squeeze(-1), segm_frames_orig[:,ids_reals,:].to(device)) # for masked LM

                if verbose: 
                    print('ids_dummies: ', ids_dummies)
                    print('T[ids_dummies]: ',T[ids_dummies])
                    print('logits_lm_with_dummy[:,ids_dummies,:]: ', logits_lm_with_dummy[:,ids_dummies,:])
                loss_lm_dummies = F.mse_loss(logits_lm_with_dummy[:,ids_dummies,:].squeeze(-1), segm_frames_orig[:,ids_dummies,:].to(device)) # for masked LM

                loss_lm = loss_lm_real*weight_loss_on_real + loss_lm_dummies*(1-weight_loss_on_real)

            else:
                logits_lm , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
                scores2 = torch.stack(scores)
                all_att = scores2.reshape(-1,scores2.shape[-1])
                loss_lm = F.mse_loss(logits_lm, masked_tokens) # for masked LM

        loss = loss_lm.float() 
        loss = loss.type_as(data)

        tmp10 = []
        for idx_tmp in range(data.shape[0]):
            if compute_loss_whole_curve:
                A = logits_lm_with_dummy.squeeze(-1)[:,ids,:][idx_tmp,:].detach().cpu().numpy().flatten()
                B=data[idx_tmp,:].detach().cpu().numpy().flatten()
            elif compute_loss_on_dummy_points:
                A = logits_lm_with_dummy.squeeze(-1)[idx_tmp,:].detach().cpu().numpy().flatten()
                B=segm_frames_orig[idx_tmp,:].detach().cpu().numpy().flatten()
            else: 
                A=logits_lm[idx_tmp,:].detach().cpu().numpy().flatten() 
                B=masked_tokens[idx_tmp,:].detach().cpu().numpy().flatten()
            r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2
        
            tmp10.append(r2_sq)
        
        mean_r2 = np.nanmean(tmp10)
        
        batch_dictionary={
            #REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            "train_r2": mean_r2,
            #optional for logging purposes
            # "log": logs
          }

        self.log("train_loss", loss.detach())
        self.log("train_loss_lm", loss_lm.detach())
        self.log("train_r2", mean_r2)
        
        #Logs
        logs={"train_loss": loss.detach(), "train_r2": mean_r2}

        if verbose:
            print('[train] plotting: ',plotting)
            print('[train] epochs_plot: ',epochs_plot)
            print('[train] self.current_epoch: ',self.current_epoch)
        
        if plotting == True and self.current_epoch%epochs_plot == 0:
            if "curve" in experiment_name or "Spirals" in experiment_name: 
                if not compute_loss_whole_curve:
                    # To make plots with the dummy points ##
                    masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                    logits_lm_with_dummy , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                
                dev_logits_lm=None
                logits_lm=None
    
                T_to_plot = T.clone().detach()
                if verbose: 
                    print('data[0]: ',data[:2])
                    print('masked_tokens[0]: ',masked_tokens[:2])
                if "Navier" in experiment_name:
                    plot_predictions(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='train')
                else:
                    plot_predictions_curves(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, None, logits_lm, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='train')

            elif "grab" in experiment_name:
                plot_predictions(data,T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='train')


        del T, P_row, P_col, segm_frames, masked_pos, masked_tokens, loss, loss_lm, tmp10, mean_r2, data, A, B #, dev_logits_lm
        
        return batch_dictionary 

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
        
    
    def on_train_epoch_start(self):
        if self.hparams.cfg["plot_predictions"] == True:
            self.hparams.cfg["plotting"] = True

    def on_train_batch_end(self, outputs, batch, batch_idx): #, dataloader_idx):
        # -- only plot for one batch, then turn off plotting
        
        if batch_idx == 0: 
            self.hparams.cfg["plotting"] = True
        else: 
            # print('[on_train_batch_end] setting plotting=False')
            self.hparams.cfg["plotting"] = False
            self.hparams.cfg["plot_training_curves"] = False
        
        
    def validation_epoch_end(self, outputs):
        self.epoch_counter+=1
        if self.epoch_counter > self.warmup:
            # Lipschitz_regularization = False #self.hparams.cfg["Lipschitz_regularization"]

            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

            self.val_loss.append(avg_loss.item())

            if self.epoch_counter == self.warmup:
                print('Warmup reached')
            if self.epoch_counter > self.warmup:
                self.save_best_model(self.path_to_save_models, self.val_loss[-1], self.epoch_counter, self.model)

            tensorboard_logs = {'val_loss': avg_loss, 
                                'step': self.current_epoch,
                               }

            # logging using tensorboard logger
            self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
            return {'log': tensorboard_logs}
        
        else:
            print('Still warmup')
            avg_loss = float('inf')
    
    def training_epoch_end(self, outputs):
        # Lipschitz_regularization = False #self.hparams.cfg["Lipschitz_regularization"]
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss, 
                            'step': self.current_epoch,
                           }
        # logging histograms
        self.custom_histogram_adder()
    
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",avg_loss,self.current_epoch)
    
        epoch_dictionary={
            'loss': avg_loss,
            'log': tensorboard_logs}

        
    def validation_step(self, batch, batch_idx):
        verbose=False
        if self.sobolev_loss_ is not None:
            torch.set_grad_enabled(True)
        # -- load variables from cfg dict
        n_pred = self.hparams.patch_sampling_cfg["num_patches_to_hide"]
        num_patches = self.hparams.patch_sampling_cfg["num_patches"]
        num_frames = self.hparams.patch_sampling_cfg["num_frames"]
        frame_size = self.hparams.cfg["frame_size"]
        patch_size = self.hparams.cfg["patch_size"]
        range_imshow = self.hparams.cfg["range_imshow"]
        model_type = self.hparams.cfg["model_type"]
        scale_gauss = self.hparams.cfg["scale_gauss"]
        penalty_orthogonality = self.hparams.cfg["penalty_orthogonality"]
        in_between_frame_init = self.hparams.patch_sampling_cfg["in_between_frame_init"]
        experiment_name = self.hparams.train_cfg["experiment_name"]
        compute_loss_whole_curve = self.hparams.train_cfg["compute_loss_whole_curve"]
        compute_loss_on_dummy_points = self.hparams.train_cfg["compute_loss_on_dummy_points"]
        if compute_loss_on_dummy_points:
            weight_loss_on_real = self.hparams.train_cfg["weight_loss_on_real"]
        plot_att_weights = self.hparams.cfg["plot_att_weights"]
        std_noise_t = self.hparams.train_cfg["std_noise_t"]
        plotting = self.hparams.cfg["plotting"]
        epochs_plot = self.hparams.cfg["plot_every_epochs"]
        inference=False
        
        
        # -- batch has data on [0] and targets on [1]
        data = batch[0]
        T_orig = batch[1][0].cpu() #Just the first item 
        if verbose: 
            print('[validation_step] data.shape: ',data.shape)
            print('[validation_step] T_orig: ',T_orig)
        
        # -- sample patch in time and space
        # -- returns patch coordinates, segment with patches, and masked positions
        T, \
        P_row, \
        P_col, \
        segm_frames, \
        masked_pos, \
        masked_tokens = patch_sampling(data, T_orig, self.hparams.patch_sampling_cfg, frame_size, patch_size)
        if self.sobolev_loss_ is not None:
            segm_frames = segm_frames.to(device).requires_grad_(True)
        if verbose: print('[val] masked_pos: ',masked_pos)
        if verbose: print('[val] masked_tokens: ',masked_tokens)
        if verbose: print('[val] masked_pos.shape: ',masked_pos.shape)
        if verbose: print('[val] T[masked_pos[0,:].long()]: ',T[masked_pos[0,:]])
        
        if model_type=='conv_encoder':
            segm_frames = segm_frames.reshape(segm_frames.shape[0],1,patch_size[0],patch_size[1])

        if verbose: print('[val] segm_frames.shape: ',segm_frames.shape)

        if plotting == True and self.current_epoch%epochs_plot == 0:
            dev_logits_lm=None
            logits_lm=None
            loss=None
            mean_r2=None
            if "Navier" in experiment_name:
                plot_predictions(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, segm_frames, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='val_segFram')
            else:
                plot_predictions_curves(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, None, logits_lm, segm_frames, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='val_segFram')
        if inference:
            dev_logits_lm=None
            logits_lm=None
            loss=None
            mean_r2=None
            plot_predictions_curves(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, dev_logits_lm, segm_frames, patch_size,self.current_epoch, loss, mean_r2, None, mode='val_segFram_inference')
        
            
        latentVar = self.model.encoder(segm_frames.type_as(data))

        if verbose: print('[val] latentVar.shape: ',latentVar.shape)

#         x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
#         mask : (B(batch_size) x S(seq_len))
        input_mask = torch.ones(latentVar.shape[0], latentVar.shape[1])
       
        mapping_size = 256
        if scale_gauss is None: 
            map_fn = None 
        else: 
            map_fn = np.random.normal(0, scale=1, size=(mapping_size, 3)) * scale_gauss
            

        if self.sobolev_loss_ is not None:
            masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
            logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
            scores2 = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
            all_att = scores2.reshape(-1,scores2.shape[-1])
            
            ids = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
            ids2 = np.argwhere(np.isin(T,T_orig,invert=True)).flatten()
            
            loss_lm = self.sobolev_loss_.evaluate__loss(y=logits_lm_with_dummy.squeeze(-1),
                                                        data=data.squeeze(-1),
                                                        x=(segm_frames),
                                                        x_fd=T.to(device),
                                                        y_0=logits_lm_with_dummy.squeeze(-1)[:,ids,:],
                                                        indexes = ids2
                                                        )
            

        else: #if it is not sobolev
            if compute_loss_whole_curve: 
                masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                scores2 = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
                all_att = scores2.reshape(-1,scores2.shape[-1])
                ids = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                
                loss_lm = F.mse_loss(logits_lm_with_dummy[:,ids,:], data) # for masked LM
            elif compute_loss_on_dummy_points:
                masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                if verbose: print('masked_pos_with_dummy.shape: ',masked_pos_with_dummy.shape)
                logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                scores2 = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
                all_att = scores2.reshape(-1,scores2.shape[-1])
                
                ids_reals = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                loss_lm_real = F.mse_loss(logits_lm_with_dummy[:,ids_reals,:].squeeze(-1), segm_frames[:,ids_reals,:].to(device)) # for masked LM

                ids_dummies = np.argwhere(np.isin(T,T_orig,invert=True)).flatten()
                loss_lm_dummies = F.mse_loss(logits_lm_with_dummy[:,ids_dummies,:].squeeze(-1), segm_frames[:,ids_dummies,:].to(device)) # for masked LM

                loss_lm = loss_lm_real*weight_loss_on_real + loss_lm_dummies*(1-weight_loss_on_real)

            else: 
                logits_lm , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
                scores2 = torch.stack(scores)
                all_att = scores2.reshape(-1,scores2.shape[-1])
                loss_lm = F.mse_loss(logits_lm, masked_tokens) # for masked LM
        
        loss = loss_lm.float() #- weight_vne*vne
        loss = loss.type_as(data)

        tmp10 = []
        for idx_tmp in range(data.shape[0]):
            if compute_loss_whole_curve:
                A = logits_lm_with_dummy.squeeze(-1)[:,ids,:][idx_tmp,:].detach().cpu().numpy().flatten()
                B=data[idx_tmp,:].detach().cpu().numpy().flatten()
            elif compute_loss_on_dummy_points:
                A = logits_lm_with_dummy.squeeze(-1)[idx_tmp,:].detach().cpu().numpy().flatten()
                B=segm_frames[idx_tmp,:].detach().cpu().numpy().flatten()
            else: 
                A=logits_lm[idx_tmp,:].detach().cpu().numpy().flatten() #Original Sept30th 2022
                B=masked_tokens[idx_tmp,:].detach().cpu().numpy().flatten()
            r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2
        
            tmp10.append(r2_sq)
        
        mean_r2 = np.nanmean(tmp10)

        self.log("val_loss", loss)
        self.log("val_loss_lm", loss_lm.detach())
        self.log("val_r2", mean_r2)
        
        #Logs
        logs={"val_loss": loss, "val_r2": mean_r2} 
        batch_dictionary={
            "val_loss": loss,
            "val_r2": mean_r2,
            "log": logs
          }
        
        plotting = self.hparams.cfg["plotting"]
        epochs_plot = self.hparams.cfg["plot_every_epochs"]
        if verbose:
            print('[val] plotting: ',plotting)
            print('[val] epochs_plot: ',epochs_plot)            
        
        if plotting == True and self.current_epoch%epochs_plot == 0:
            if "curve" in experiment_name or "Spirals" in experiment_name: 
                if not compute_loss_whole_curve:
                    masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                    logits_lm_with_dummy , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                
                dev_logits_lm=None
                logits_lm=None
                if "Navier" in experiment_name:
                    plot_predictions(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='val')
                else:
                    plot_predictions_curves(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, dev_logits_lm, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='val')

                if plot_att_weights:
                    plot_attention_weights(scores,self.current_epoch, self.logger.log_dir, mode='val')
   
            elif "grab" in experiment_name:
                plot_predictions(data,T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='val')
    
            
        del T, P_row, P_col, segm_frames, masked_pos, masked_tokens, loss, latentVar
     
            
        if self.hparams.cfg["plot_training_curves"] == True: 
            plot_training_curves(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size, 
                     self.current_epoch, loss, mean_r2, self.logger.log_dir)
            
        return batch_dictionary  
    

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.hparams.train_cfg["learning_rate"],
                         weight_decay=self.hparams.train_cfg["weight_decay"])
        return optimizer
    

class BERT_LinearEncoder(nn.Module):
    "Bert Model : Masked LM and next sentence classification"
    def __init__(self, cfg, n_labels=1):
        super(BERT_LinearEncoder, self).__init__()
        self.transformer = ContSpaceTime.Transformer(cfg)
        self.activ1 = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(cfg["p_drop_hidden"])
        self.activ2 = ContSpaceTime.gelu
        self.norm = ContSpaceTime.LayerNorm(cfg)
        self.beta_Lipschitz=0.8
        self.gamma_Lipschitz=0.01
        
        dim_images = cfg["patch_size"][0]* cfg["patch_size"][1] #In this case, the patches are square, so one dimension is enough to determine the final image dim_image
    
        self.linear = nn.Linear(cfg["dim"], cfg["dim"])
        self.decoder = nn.Linear(cfg["dim"], dim_images, bias=False)
        if cfg["operation_with_pos_encoding"] == "concatenate":
            self.encoder = nn.Linear(dim_images, int(cfg["dim"]/2), bias=False)
        elif cfg["operation_with_pos_encoding"] == "sum":  
            self.encoder = nn.Linear(dim_images, cfg["dim"], bias=False)
        nn.init.xavier_normal_(self.decoder.weight, gain=nn.init.calculate_gain('tanh'))
