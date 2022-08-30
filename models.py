import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, mean_squared_error
import functools
import warnings

from augmentations import *
from utils import *


def _set_augmentations(augs_params):    
    aug_string = augs_params["augmentation_string"].lower()
    augs = aug_string.split(",")
    set_augs = ["invert_seqs", "invert_rc_seqs", "delete_seqs", "roll", "roll_seqs", "insert", "insert_seqs", "rc", "mutate_seqs", "noise_gauss"]
    
    assert all(aug in set_augs for aug in augs), "unrecognized augmentation in user-defined AUGMENTATION_STRING"
    assert not ("invert_seqs" in augs and "invert_rc_seqs" in augs), "cannot have both \"invert_seqs\" and \"invert_rc_seqs\" in user-defined AUGMENTATION_STRING"
    assert not ("insert" in augs and "insert_seqs" in augs), "cannot have both \"insert\" and \"insert_seqs\" in user-defined AUGMENTATION_STRING"
    assert not ("roll" in augs and "roll_seqs" in augs), "cannot have both \"roll\" and \"roll_seqs\" in AUGMENTATION_STRING"
    if augs_params["sample_augs_num"] is not None:
        assert 0 < augs_params["sample_augs_num"] <= len(augs), "cannot have user-defined SAMPLE_AUGS_NUM greater than the total number of augmentations in user-defined AUGMENTATION_STRING"
        
    augs_order = {"invert_seqs" : 6, # invert first to ensure non-random, meaningful sequence is inverted
                  "invert_rc_seqs" : 6,
                  "delete_seqs" : 5, # delete second to also ensure meaningful sequence deleted
                  "roll" : 4,
                  "roll_seqs" : 4, 
                  "insert" : 3, # do insertion after roll to preserve flanking random sequences
                  "insert_seqs" : 3, 
                  "rc" : 2, 
                  "mutate_seqs" : 1,
                  "noise_gauss" : 0 } # add Gaussian noise last to "mutate" all one-hot positions
    augs_sorted = sorted(augs, key=lambda d: augs_order[d])
    
    set_augs_funcs = {"invert_seqs" : functools.partial(aug_invert_seqs, 
                                                        invert_min=augs_params["invert_min"], 
                                                        invert_max=augs_params["invert_max"]),
                      "invert_rc_seqs" : functools.partial(aug_invert_rc_seqs, 
                                                        invert_min=augs_params["invert_min"], 
                                                        invert_max=augs_params["invert_max"]),
                      "delete_seqs" : functools.partial(aug_delete_seqs, 
                                                        delete_min=augs_params["delete_min"], 
                                                        delete_max=augs_params["delete_max"]),
                      "roll" : functools.partial(aug_roll, 
                                                 shift_min=augs_params["shift_min"], 
                                                 shift_max=augs_params["shift_max"]), 
                      "roll_seqs" : functools.partial(aug_roll_seqs, 
                                                      shift_min=augs_params["shift_min"], 
                                                      shift_max=augs_params["shift_max"]),
                      "insert" : functools.partial(aug_insert, 
                                                   insert_min=augs_params["insert_min"], 
                                                   insert_max=augs_params["insert_max"]),
                      "insert_seqs" : functools.partial(aug_insert_seqs, 
                                                        insert_min=augs_params["insert_min"],
                                                        insert_max=augs_params["insert_max"]),
                      "rc" : functools.partial(aug_rc, rc_prob=augs_params["rc_prob"]),
                      "mutate_seqs" : functools.partial(aug_mutate_seqs, mutate_frac=augs_params["mutate_frac"]),
                      "noise_gauss" : functools.partial(aug_noise_gauss, std=augs_params["std"])}
    
    if augs_params["sample_augs_hard"] and (augs_params["sample_augs_num"] == len(augs) or augs_params["sample_augs_num"] == None): # i.e., if using (default) full augmentation strategy, return composed full augmentation function
        augs_funcs_1 = [set_augs_funcs[aug] for aug in augs_sorted if aug != "rc"] # augs_funcs_1 will never contain aug_rc()
        augs_funcs_2 = [set_augs_funcs[aug] for aug in augs_sorted]
        augmentation_function_1 = compose(*augs_funcs_1) if augs_funcs_1 else lambda x : x
        augmentation_function_2 = compose(*augs_funcs_2)
        return augmentation_function_1, augmentation_function_2
    else: # if using stochastic augmentation strategy, return aug. partials with padding 
        pad_end_partial = functools.partial(aug_pad_end, insert_max=augs_params["insert_max"])
        augs_funcs = []
        for aug in augs_sorted:
            if "insert" not in aug_string or "insert" in aug:
                func = set_augs_funcs[aug]
            elif aug == "noise_gauss": # and (by the logical structure) "insert" is in aug_string
                func = compose(*[set_augs_funcs[aug], pad_end_partial]) # for Gaussian noise aug., pad *before* augmenting
            else: # and, by implication (of the logical structure), "insert" is in aug_string
                func = compose(*[pad_end_partial, set_augs_funcs[aug]]) # for all other augs., pad *after* augmenting
            augs_funcs.append(func)
        return augs_funcs


def _hash_all_combs_of_augs(augs_params):    
    aug_string = augs_params["augmentation_string"].lower()
    augs = aug_string.split(",")
    augs_order = {"invert_seqs" : 6, # invert first to ensure non-random, meaningful sequence is inverted
                  "invert_rc_seqs" : 6,
                  "delete_seqs" : 5, # delete second to also ensure meaningful sequence deleted
                  "roll" : 4,
                  "roll_seqs" : 4, 
                  "insert" : 3, # do insertion after roll to preserve flanking random sequences
                  "insert_seqs" : 3, 
                  "rc" : 2, 
                  "mutate_seqs" : 1,
                  "pad_end" : 0.5, # if random padding needed, do after all other augs but before Gaussian noise
                  "noise_gauss" : 0 } # add Gaussian noise last to "mutate" all one-hot positions
    augs_sorted = sorted(augs, key=lambda d: augs_order[d])
    
    set_augs_funcs = {"invert_seqs" : functools.partial(aug_invert_seqs, 
                                                        invert_min=augs_params["invert_min"], 
                                                        invert_max=augs_params["invert_max"]),
                      "invert_rc_seqs" : functools.partial(aug_invert_rc_seqs, 
                                                        invert_min=augs_params["invert_min"], 
                                                        invert_max=augs_params["invert_max"]),
                      "delete_seqs" : functools.partial(aug_delete_seqs, 
                                                        delete_min=augs_params["delete_min"], 
                                                        delete_max=augs_params["delete_max"]),
                      "roll" : functools.partial(aug_roll, 
                                                 shift_min=augs_params["shift_min"], 
                                                 shift_max=augs_params["shift_max"]), 
                      "roll_seqs" : functools.partial(aug_roll_seqs, 
                                                      shift_min=augs_params["shift_min"], 
                                                      shift_max=augs_params["shift_max"]),
                      "insert" : functools.partial(aug_insert, 
                                                   insert_min=augs_params["insert_min"], 
                                                   insert_max=augs_params["insert_max"]),
                      "insert_seqs" : functools.partial(aug_insert_seqs, 
                                                        insert_min=augs_params["insert_min"],
                                                        insert_max=augs_params["insert_max"]),
                      "rc" : functools.partial(aug_rc, rc_prob=augs_params["rc_prob"]),
                      "mutate_seqs" : functools.partial(aug_mutate_seqs, mutate_frac=augs_params["mutate_frac"]),
                      "pad_end" : functools.partial(aug_pad_end, insert_max=augs_params["insert_max"]),
                      "noise_gauss" : functools.partial(aug_noise_gauss, std=augs_params["std"])}
    
    allcombs = list(powerset_without_empty(range(len(augs_sorted))))
    allfuncs = []
    for comb in allcombs:
        parts_string = ",".join([augs_sorted[i] for i in comb])
        parts = [augs_sorted[i] for i in comb] if "insert" not in aug_string or "insert" in parts_string else [augs_sorted[i] for i in comb] + ["pad_end"]
        parts_sorted = sorted(parts, key=lambda d: augs_order[d]) # contains "pad_end" at appropriate index
        allfuncs.append( compose(*[set_augs_funcs[aug] for aug in parts_sorted]) )
    
    return {key : value for key, value in zip(allcombs, allfuncs)}


class SupervisedModel(LightningModule):
    """supervised learning model or supervised transfer learning model without data augmentations
        
        Parameters:
            model_untrained: untrained supervised model *OR* untrained supervised 
                transfer learning model into which trained first-layer convolutional filters 
                from GRIM have been inserted (see supervised.py); should be an instance 
                of a class inheriting from torch.nn.Module
            loss_criterion: loss criterion to use--should be a function, e.g. nn.BCELoss()
    """
    def __init__(self, model_untrained, loss_criterion, optimizers_configured=None):
        super().__init__()
        self.model = model_untrained
        self.criterion = loss_criterion # should be a function--e.g., nn.BCELoss()
        self.optimizers_configured = optimizers_configured
    
    def configure_optimizers(self):
        if self.optimizers_configured is None: # i.e., if no optimizer configuration given by user
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=1e-3, weight_decay=1e-6)
            return optimizer
        else: # i.e., an optimizer configuration *was* given by user
            return self.optimizers_configured
    
    def forward(self, x):
        y_hat = self.model(x)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch # assumes batch is tuple of (x, y)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
        # with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
        #     warnings.simplefilter("ignore")
        #     aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) )        
        # self.log('train_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('train_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
        # with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
        #     warnings.simplefilter("ignore")
        #     aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) )        
        # self.log('val_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if isinstance(self.criterion, torch.nn.modules.loss.BCELoss):
            auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
            with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
                warnings.simplefilter("ignore")
                aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) ) 
            self.log('test_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('test_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
        elif isinstance(self.criterion, torch.nn.modules.loss.MSELoss):
            for i in range(y.shape[-1]):    
                mse_i = mean_squared_error(y[:,i].cpu(), y_hat[:,i].cpu()).item()
                r_i = stats.pearsonr(y[:,i].cpu(), y_hat[:,i].cpu())[0]
                rho_i = stats.spearmanr(y[:,i].cpu(), y_hat[:,i].cpu())[0]
                self.log('test_mse_'+str(i), mse_i, on_step=False, on_epoch=True, prog_bar=True)
                self.log('test_pearson_r_'+str(i), r_i, on_step=False, on_epoch=True, prog_bar=True)
                self.log('test_spearman_rho_'+str(i), rho_i, on_step=False, on_epoch=True, prog_bar=True)


class SupervisedModelWithAugmentation(LightningModule):
    """supervised learning model or supervised transfer learning model with data augmentation
        
        Parameters:
            model_untrained: untrained supervised model *OR* untrained supervised 
                transfer learning model into which trained first-layer convolutional filters 
                from GRIM have been inserted (see supervised.py); should be an instance 
                of a class inheriting from torch.nn.Module
            loss_criterion: loss criterion to use--should be a function, e.g. nn.BCELoss()
            
            augmentation_string (optional): string specifying augmentations to use, comma delimited; 
                possible augmentations are \"invert_seqs\", \"invert_rc_seqs\", \"delete_seqs\", 
                \"roll\", \"roll_seqs\", \"insert\", \"insert_seqs\", \"rc\", \"mutate_seqs\", 
                \"noise_gauss\" (default: "noise_gauss,rc,insert_seqs")
            invert_min (optional): in inversion augmentations, minimum length of inversion
                (default: 0)
            invert_max (optional): in inversion augmentations, maximum length of inversion
                (default: 50)
            delete_min (optional): in deletion augmentations, minimum length of deletion
                (default: 0)
            delete_max (optional): in deletion augmentations, maximum length of deletion
                (default: 25)
            shift_min (optional): in roll augmentations, minimum number of places by which 
                position can be shifted (default: 0)
            shift_max (optional): in roll augmentations, maximum number of places by which 
                position can be shifted (default: 25)
            insert_min (optional): in insertion augmentations, minimum length of insertion 
                (default: 0)
            insert_max (optional): in insertion augmentations, maximum length of insertion 
                (default: 50)
            rc_prob (optional): in reverse complement augmentation, probability for each sequence 
                to be \"mutated\" to its reverse complement (default: 0.5)
            mutate_frac (optional): in random mutation augmentation, fraction of each sequence's 
                nucleotides to mutate (default: 0.1)
            std (optional): in Gaussian noise addition augmentation, standard deviation of 
                Gaussian distribution from which noise is drawn (default: 0.2)
                
    """
    def __init__(self, model_untrained, loss_criterion, optimizers_configured=None, 
                 augmentation_string="noise_gauss,rc,insert_seqs",
                 invert_min=0, invert_max=50, delete_min=0, delete_max=25,
                 shift_min=0, shift_max=25, insert_min=0, insert_max=50, 
                 rc_prob=0.5, mutate_frac=0.1, std=0.2):
        super().__init__()
        self.model = model_untrained
        self.criterion = loss_criterion # should be a function--e.g., nn.BCELoss()
        self.optimizers_configured = optimizers_configured
        
        self.augmentation_string = augmentation_string
        self.invert_min = invert_min
        self.invert_max = invert_max
        self.delete_min = delete_min
        self.delete_max = delete_max
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.insert_min = insert_min
        self.insert_max = insert_max
        self.rc_prob = rc_prob
        self.mutate_frac = mutate_frac
        self.std = std
        
        # Set augmentation functions
        augs_params = {"augmentation_string" : self.augmentation_string,
                       "sample_augs_hard" : True, "sample_augs_num" : None,
                       "invert_min" : self.invert_min, "invert_max" : self.invert_max, 
                       "delete_min" : self.delete_min, "delete_max" : self.delete_max,
                       "shift_min" : self.shift_min, "shift_max" : self.shift_max,
                       "insert_min" : self.insert_min, "insert_max" : self.insert_max,
                       "rc_prob" : self.rc_prob, "mutate_frac" : self.mutate_frac, "std" : self.std}
        _, self.augment = _set_augmentations(augs_params)
    
    def configure_optimizers(self):
        if self.optimizers_configured is None: # i.e., if no optimizer configuration given by user
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=1e-3, weight_decay=1e-6)
            return optimizer
        else: # i.e., an optimizer configuration *was* given by user
            return self.optimizers_configured
    
    def forward(self, x):
        y_hat = self.model(x)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch # assumes batch is tuple of (x, y)
        y_hat = self( self.augment(x) )
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
        # with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
        #     warnings.simplefilter("ignore")
        #     aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) )
        # self.log('train_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('train_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch 
        x_padded = aug_pad_end(x, self.insert_max) if "insert" in self.augmentation_string else x
        y_hat = self( x_padded )
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
        # with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
        #     warnings.simplefilter("ignore")
        #     aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) )        
        # self.log('val_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_padded = aug_pad_end(x, self.insert_max) if "insert" in self.augmentation_string else x
        y_hat = self( x_padded )
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if isinstance(self.criterion, torch.nn.modules.loss.BCELoss):
            auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
            with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
                warnings.simplefilter("ignore")
                aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) ) 
            self.log('test_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('test_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
        elif isinstance(self.criterion, torch.nn.modules.loss.MSELoss):
            for i in range(y.shape[-1]):    
                mse_i = mean_squared_error(y[:,i].cpu(), y_hat[:,i].cpu()).item()
                r_i = stats.pearsonr(y[:,i].cpu(), y_hat[:,i].cpu())[0]
                rho_i = stats.spearmanr(y[:,i].cpu(), y_hat[:,i].cpu())[0]
                self.log('test_mse_'+str(i), mse_i, on_step=False, on_epoch=True, prog_bar=True)
                self.log('test_pearson_r_'+str(i), r_i, on_step=False, on_epoch=True, prog_bar=True)
                self.log('test_spearman_rho_'+str(i), rho_i, on_step=False, on_epoch=True, prog_bar=True)


class SupervisedModelWithPadding(LightningModule):
    """supervised learning model or supervised transfer learning model *without*
        data augmentations but *with* random padding of input sequences (along length dimension);
        these modules are typically used for fine-tuning of supervised models previously
        trained with insertion augmentations, which expect input sequences of
        length (L + insert_max) rather than the standard length L
        
        Parameters:
            model_untrained: untrained supervised model *OR* untrained supervised 
                transfer learning model into which trained first-layer convolutional filters 
                from GRIM have been inserted (see supervised.py); should be an instance 
                of a class inheriting from torch.nn.Module
            loss_criterion: loss criterion to use--should be a function, e.g. nn.BCELoss()
            
            insert_max (optional): in insertion augmentations, maximum length of insertion 
                (default: 50)
    """
    def __init__(self, model_untrained, loss_criterion, optimizers_configured=None,
                 insert_max=50):
        super().__init__()
        self.model = model_untrained
        self.criterion = loss_criterion # should be a function--e.g., nn.BCELoss()
        self.optimizers_configured = optimizers_configured
        self.insert_max = insert_max
    
    def configure_optimizers(self):
        if self.optimizers_configured is None: # i.e., if no optimizer configuration given by user
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=1e-5, weight_decay=1e-6)
            return optimizer
        else: # i.e., an optimizer configuration *was* given by user
            return self.optimizers_configured
    
    def forward(self, x):
        x_padded = aug_pad_end(x, self.insert_max) # if "insert" in self.augmentation_string else x
        y_hat = self.model(x_padded)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch # assumes batch is tuple of (x, y)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
        # with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
        #     warnings.simplefilter("ignore")
        #     aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) )        
        # self.log('train_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('train_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
        # with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
        #     warnings.simplefilter("ignore")
        #     aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) )        
        # self.log('val_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if isinstance(self.criterion, torch.nn.modules.loss.BCELoss):
            auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
            with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
                warnings.simplefilter("ignore")
                aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) ) 
            self.log('test_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('test_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
        elif isinstance(self.criterion, torch.nn.modules.loss.MSELoss):
            for i in range(y.shape[-1]):    
                mse_i = mean_squared_error(y[:,i].cpu(), y_hat[:,i].cpu()).item()
                r_i = stats.pearsonr(y[:,i].cpu(), y_hat[:,i].cpu())[0]
                rho_i = stats.spearmanr(y[:,i].cpu(), y_hat[:,i].cpu())[0]
                self.log('test_mse_'+str(i), mse_i, on_step=False, on_epoch=True, prog_bar=True)
                self.log('test_pearson_r_'+str(i), r_i, on_step=False, on_epoch=True, prog_bar=True)
                self.log('test_spearman_rho_'+str(i), rho_i, on_step=False, on_epoch=True, prog_bar=True)


class SupervisedModelWithStochasticAugmentation(LightningModule):
    """supervised learning model or supervised transfer learning model with data augmentation
            applied stochastically to each sequence randomly
        
        Parameters:
            model_untrained: untrained supervised model *OR* untrained supervised 
                transfer learning model into which trained first-layer convolutional filters 
                from GRIM have been inserted (see supervised.py); should be an instance 
                of a class inheriting from torch.nn.Module
            loss_criterion: loss criterion to use--should be a function, e.g. nn.BCELoss()
            
            augmentation_string (optional): string specifying augmentations to use, comma delimited,
                from which one augmentation will be randomly applied (independently for each epoch)
                to each sequence in the training set during training; possible augmentations are 
                \"invert_seqs\", \"invert_rc_seqs\", \"delete_seqs\", \"roll\", \"roll_seqs\", 
                \"insert\", \"insert_seqs\", \"rc\", \"mutate_seqs\", \"noise_gauss\" 
                (default: "noise_gauss,rc,insert_seqs")
            invert_min (optional): in inversion augmentations, minimum length of inversion
                (default: 0)
            invert_max (optional): in inversion augmentations, maximum length of inversion
                (default: 50)
            delete_min (optional): in deletion augmentations, minimum length of deletion
                (default: 0)
            delete_max (optional): in deletion augmentations, maximum length of deletion
                (default: 25)
            shift_min (optional): in roll augmentations, minimum number of places by which 
                position can be shifted (default: 0)
            shift_max (optional): in roll augmentations, maximum number of places by which 
                position can be shifted (default: 25)
            insert_min (optional): in insertion augmentations, minimum length of insertion 
                (default: 0)
            insert_max (optional): in insertion augmentations, maximum length of insertion 
                (default: 50)
            rc_prob (optional): in reverse complement augmentation, probability for each sequence 
                to be \"mutated\" to its reverse complement (default: 0.5)
            mutate_frac (optional): in random mutation augmentation, fraction of each sequence's 
                nucleotides to mutate (default: 0.1)
            std (optional): in Gaussian noise addition augmentation, standard deviation of 
                Gaussian distribution from which noise is drawn (default: 0.2)
                
    """
    def __init__(self, model_untrained, loss_criterion, optimizers_configured=None, 
                 augmentation_string="noise_gauss,rc,insert_seqs",
                 sample_augs_hard=True, sample_augs_num=None,
                 invert_min=0, invert_max=50, delete_min=0, delete_max=25,
                 shift_min=0, shift_max=25, insert_min=0, insert_max=50, 
                 rc_prob=0.5, mutate_frac=0.1, std=0.2):
        super().__init__()
        self.model = model_untrained
        self.criterion = loss_criterion # should be a function--e.g., nn.BCELoss()
        self.optimizers_configured = optimizers_configured
        
        self.augmentation_string = augmentation_string
        self.invert_min = invert_min
        self.invert_max = invert_max
        self.delete_min = delete_min
        self.delete_max = delete_max
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.insert_min = insert_min
        self.insert_max = insert_max
        self.rc_prob = rc_prob
        self.mutate_frac = mutate_frac
        self.std = std
        
        # Set augmentation functions and parameters
        augs_params = {"augmentation_string" : self.augmentation_string,
                       "sample_augs_hard" : sample_augs_hard, "sample_augs_num" : sample_augs_num,
                       "invert_min" : self.invert_min, "invert_max" : self.invert_max, 
                       "delete_min" : self.delete_min, "delete_max" : self.delete_max,
                       "shift_min" : self.shift_min, "shift_max" : self.shift_max,
                       "insert_min" : self.insert_min, "insert_max" : self.insert_max,
                       "rc_prob" : self.rc_prob, "mutate_frac" : self.mutate_frac, "std" : self.std}
        self.aug_funcs = _set_augmentations(augs_params)
        self.hashmap_combs_augfuncs = _hash_all_combs_of_augs(augs_params)
        
        self.num_augs = len(self.augmentation_string.split(","))
        self.sample_augs_hard = sample_augs_hard
        self.sample_augs_num = self.num_augs if sample_augs_num == None else sample_augs_num
        
        self.sample_num_augs_func = (lambda num : self.sample_augs_num * np.ones((num,), dtype=int)) if self.sample_augs_hard else (lambda num : np.random.randint(1, self.sample_augs_num + 1, (num,))) # function to determine number of augmentations to sample for each sequence in batch of x
        
    def configure_optimizers(self):
        if self.optimizers_configured is None: # i.e., if no optimizer configuration given by user
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=1e-3, weight_decay=1e-6)
            return optimizer
        else: # i.e., an optimizer configuration *was* given by user
            return self.optimizers_configured
    
    def forward(self, x):
        y_hat = self.model(x)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch # assumes batch is tuple of (x, y)
        sample_nums = self.sample_num_augs_func( x.shape[0] )
        aug_combs = [ tuple(sorted(np.random.choice(self.num_augs, sample, replace=False))) for sample in sample_nums ]
        x_aug = torch.cat( [self.hashmap_combs_augfuncs[aug_comb](torch.unsqueeze(seq, dim=0)) for aug_comb, seq in zip(aug_combs, x)] ) # stochastic augmentation process
        y_hat = self( x_aug )
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
        # with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
        #     warnings.simplefilter("ignore")
        #     aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) )
        # self.log('train_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('train_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch 
        x_padded = aug_pad_end(x, self.insert_max) if "insert" in self.augmentation_string else x
        y_hat = self( x_padded )
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
        # with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
        #     warnings.simplefilter("ignore")
        #     aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) )        
        # self.log('val_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_padded = aug_pad_end(x, self.insert_max) if "insert" in self.augmentation_string else x
        y_hat = self( x_padded )
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if isinstance(self.criterion, torch.nn.modules.loss.BCELoss):
            auroc = np.nanmean( calculate_auroc(y.cpu().numpy(), y_hat.cpu().numpy()) ) 
            with warnings.catch_warnings(): # to catch RuntimeWarnings from division by zero
                warnings.simplefilter("ignore")
                aupr = np.nanmean( average_precision_score(y.cpu().numpy(), y_hat.cpu().numpy(), average=None) ) 
            self.log('test_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('test_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True)
        elif isinstance(self.criterion, torch.nn.modules.loss.MSELoss):
            for i in range(y.shape[-1]):    
                mse_i = mean_squared_error(y[:,i].cpu(), y_hat[:,i].cpu()).item()
                r_i = stats.pearsonr(y[:,i].cpu(), y_hat[:,i].cpu())[0]
                rho_i = stats.spearmanr(y[:,i].cpu(), y_hat[:,i].cpu())[0]
                self.log('test_mse_'+str(i), mse_i, on_step=False, on_epoch=True, prog_bar=True)
                self.log('test_pearson_r_'+str(i), r_i, on_step=False, on_epoch=True, prog_bar=True)
                self.log('test_spearman_rho_'+str(i), rho_i, on_step=False, on_epoch=True, prog_bar=True)
