import torch
#from .equiformer import * 
from .consistency import * 
#from .en_transformer import * 
from .Equiformerv2 import *

import pickle,os,tempfile, shutil, zipfile, time, math, tqdm 
from datetime import datetime 
from ..comparm import * 
from ..utils.utils_torch import *
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from ..graphs.datasets import *
from tqdm import tqdm 
from torch import distributed as dist
from .modules import * 
from .modules_MHA import *
from .modules_ctrl_nocgraph import * 

class TreeInvent_Model:
    def __init__(self,local_rank=None,jobs='coords',**kwargs):
        epochs=kwargs.get('start')
        self.local_rank=local_rank
        self.batchsize=FGP.batchsize*FGP.accsteps
        self.device=FGP.device
        self.jobs=jobs

        if "modelname" not in kwargs:
            self.mode="train"
            self.modelname='TreeInvent_Model'
        else:
            self.mode='test'
            self.modelname=kwargs.get('modelname')
            
        self.__reset()
        if not os.path.exists(f'./{self.modelname}/model'):
            os.system(f'mkdir -p ./{self.modelname}/model')        
        
        self.__build_model()
        
        self.load_cpkt(f'./{self.modelname}/model')

        if epochs:
            self.epochs=epochs 
        return
    
    def __build_model(self):
        if self.local_rank is not None:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        if 'coords' in self.jobs: 
            self.conf_online_model=EquiformerV2(
                    feat_dim=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                    edge_dim=len(FGP.bond_types)+1,
                    num_layers=FGP.conf_depth,
                    num_heads=8,
                    edge_channels=FGP.conf_edge_channels,
                    sphere_channels=FGP.conf_sphere_channels,
                    ffn_hidden_channels=FGP.conf_ffn_hidden_channels,
                    attn_alpha_channels=FGP.conf_attn_alpha_channels,
                    attn_value_channels=FGP.conf_attn_value_channels,
                    num_distance_basis=FGP.conf_num_distance_basis,
                    num_sphere_samples=FGP.conf_num_sphere_samples,
                    target="L2"
                )
            
            self.conf_ema_model=EquiformerV2(
                    feat_dim=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                    edge_dim=len(FGP.bond_types)+1,
                    num_layers=FGP.conf_depth,
                    num_heads=8,
                    edge_channels=FGP.conf_edge_channels,
                    sphere_channels=FGP.conf_sphere_channels,
                    ffn_hidden_channels=FGP.conf_ffn_hidden_channels,
                    attn_alpha_channels=FGP.conf_attn_alpha_channels,
                    attn_value_channels=FGP.conf_attn_value_channels,
                    num_distance_basis=FGP.conf_num_distance_basis,
                    num_sphere_samples=FGP.conf_num_sphere_samples,
                    target="L2"
                )

            self.conf_online_model=self.__to_model_device(self.conf_online_model)
            self.conf_ema_model=self.__to_model_device(self.conf_ema_model)
        
            self.consistency_training=ConsistencyTraining(
                sigma_min=FGP.sigma_min,
                sigma_max=FGP.sigma_max,
                sigma_data=FGP.sigma_data,
                rho=FGP.rho,
                initial_timesteps=FGP.initial_timesteps,
                final_timesteps=FGP.final_timesteps
                )
        
            self.consistency_sampling_and_editing = ConsistencySamplingAndEditing(
                        sigma_min = FGP.sigma_min, # minimum std of noise
                        sigma_data = FGP.sigma_data, # std of the data
                        )
            
        if 'nadd' in self.jobs:
            self.node_add_model=self.__to_model_device(Node_adder_ctrl())

        if 'rgen' in self.jobs:
            self.ring_gen_model=self.__to_model_device(Ring_gener_ctrl()) 
            
        if 'nconn' in self.jobs:
            self.node_conn_model=self.__to_model_device(Node_connect_ctrl())

        if 'nint' in self.jobs:
            self.node_int_model=self.__to_model_device(Node_int_ctrl())
        return  
    
    def load_cpkt(self,dirpath):
        if 'coords' in self.jobs and FGP.load_dict['coords'] is not None:
            self.conf_online_model=self.load_params_into_models(self.conf_online_model,
                                                                cpkt_path=f"{FGP.load_dict['coords']}")
            print ("Load conf online model successfully!")

        if 'coords' in self.jobs and FGP.load_dict['ema'] is not None:
            self.conf_ema_model=self.load_params_into_models(self.conf_ema_model,
                                                             cpkt_path=f"{FGP.load_dict['ema']}")
            print ("Load conf ema model successfully!")

        if 'nadd' in self.jobs and FGP.load_dict['nadd'] is not None: 
            self.node_add_model=self.load_params_into_models(self.node_add_model,
                                                             cpkt_path=f"{FGP.load_dict['nadd']}")
            print ("Load nadd model successfully!")
            
        if 'rgen' in self.jobs and FGP.load_dict['rgen'] is not None:
            self.ring_gen_model=self.load_params_into_models(self.ring_gen_model,
                                                             cpkt_path=f"{FGP.load_dict['rgen']}")
            print ("Load rgen model successfully!")
            
        if 'nconn' in self.jobs and FGP.load_dict['nconn'] is not None:
            self.node_conn_model=self.load_params_into_models(self.node_conn_model,
                                                              cpkt_path=f"{FGP.load_dict['nconn']}")
            print ("Load nconn model successfully!")
        
        if 'nint' in self.jobs and FGP.load_dict['nint'] is not None:
            self.node_int_model=self.load_params_into_models(self.node_int_model,
                                                             cpkt_path=f"{FGP.load_dict['nint']}")
            print ("Load nint model successfully!")    
            
        return 

    def load_params_into_models(self,model,cpkt_path):
        if os.path.exists(cpkt_path):
            modelcpkt=torch.load(cpkt_path,map_location=torch.device('cpu'))
            para_sd=load_state_dict_to_single(modelcpkt)
            model.load_state_dict(para_sd,strict=False) # strict=False in the case of ctrlnet, which only a part of the model should be loaded when training from backbone model.
        else:
            print (f"Cannot find {cpkt_path} !")
            #exit()
        return model 
    
    def IC_Loss(self,pred,target,zbs,zas,zds,zbmasks,zamasks,zdmasks,gmasks):
        pred_bonddis,pred_angle,pred_dihedral=xyz2ics_v2(pred,zbs,zas,zds)
        target_bonddis,target_angle,target_dihedral=xyz2ics_v2(target,zbs,zas,zds)
        
        pred_dismat=torch.cdist(pred,pred,compute_mode='donot_use_mm_for_euclid_dist')
        target_dismat=torch.cdist(target,target,compute_mode='donot_use_mm_for_euclid_dist')
        #print (pred_dismat.shape,target_dismat.shape,gmasks.shape)
        gmasks_2D=(gmasks.unsqueeze(-1)*(gmasks.unsqueeze(-1).permute(0,2,1))).bool()
        if torch.sum(gmasks_2D)>0:
            #print (gmasks_2D.shape)
            loss_dismat=F.mse_loss(pred_dismat[gmasks_2D],target_dismat[gmasks_2D])
        else:
            loss_dismat=torch.tensor(0.0).to(pred.device)
            
        if torch.sum(zbmasks)>0:
            loss_bonddis=F.mse_loss(pred_bonddis[zbmasks],target_bonddis[zbmasks])
        else:
            loss_bonddis=torch.tensor(0.0).to(pred.device)
            
        if torch.sum(zamasks)>0:
            loss_angle=F.mse_loss(pred_angle[zamasks],target_angle[zamasks])
        else:
            loss_angle=torch.tensor(0.0).to(pred.device)
        
        if torch.sum(zdmasks)>0:
            dihedral_diff=torch.abs(pred_dihedral[zdmasks]-target_dihedral[zdmasks])
            dihedral_diff=torch.where(dihedral_diff>math.pi,math.pi*2-dihedral_diff,dihedral_diff)
            loss_dihedral=torch.mean(torch.square(dihedral_diff))
        else:
            loss_dihedral=torch.tensor(0.0).to(pred.device)
            
        return loss_dismat,loss_bonddis,loss_angle,loss_dihedral
    
    def Complex_Coord_Gen_Datas_to_gpu(self,Datas):
        c_feats=self.__to_device(Datas["C_Feats"].float())
        c_atoms=self.__to_device(Datas["C_Atoms"].float())
        c_adjs_mat=self.__to_device(Datas["C_Adjs_Mat"].float())
        c_coords=self.__to_device(Datas["C_Coords"].float())
        c_masks=self.__to_device(Datas["C_Masks"])
        c_fix_labels=self.__to_device(Datas["C_Fix_Masks"].float())
        
        c_flexible_labels=self.__to_device(Datas["C_Flexible_Masks"].float())
        (l_zbs,l_zas,l_zds)=(self.__to_device(Datas[key]) for key in ["L_Zbs","L_Zas","L_Zds"])
        (l_zbmasks,l_zamasks,l_zdmasks)=(self.__to_device(Datas[key].bool()) for key in ["L_Zbmask","L_Zamask","L_Zdmask"])
        (p_zbs,p_zas,p_zds)=(self.__to_device(Datas[key]) for key in ["P_Zbs","P_Zas","P_Zds"])
        (p_zbmasks,p_zamasks,p_zdmasks)=(self.__to_device(Datas[key].bool()) for key in ["P_Zbmask","P_Zamask","P_Zdmask"]) 
        c_pocket_labels=self.__to_device(Datas["C_Pocket_Labels"])
        c_ligand_labels=self.__to_device(Datas["C_Ligand_Labels"])
        step_ids=self.__to_device(Datas["Step_id"])
        
        return c_atoms,c_feats,c_adjs_mat,c_coords,c_fix_labels,c_flexible_labels,c_masks,\
            l_zbs,l_zas,l_zds,l_zbmasks,l_zamasks,l_zdmasks,\
            p_zbs,p_zas,p_zds,p_zbmasks,p_zamasks,p_zdmasks,\
            c_pocket_labels,c_ligand_labels,step_ids

    
    def Train_Complex_Conf_Step(self,Datas,step_id,mode='train'):
        c_atoms,c_feats,c_adjs_mat,c_coords,c_fix_labels,c_flexible_labels,c_masks,\
            l_zbs,l_zas,l_zds,l_zbmasks,l_zamasks,l_zdmasks,\
                p_zbs,p_zas,p_zds,p_zbmasks,p_zamasks,p_zdmasks,\
                    c_pocket_labels,c_ligand_labels=self.Complex_Coord_Gen_Datas_to_gpu(Datas)
        self.optim.zero_grad()

        total_loss=0
        total_loss_dismat=0
        total_loss_bonddis=0
        total_loss_angle=0
        total_loss_dihedral=0
        total_loss_xyz=0
        total_loss_pl=0

        num=0
        for i in range(FGP.accsteps):
            num+=1
            if len(c_feats[i*FGP.batchsize:(i+1)*FGP.batchsize])>0:
                predicted,target=self.consistency_training(self.conf_online_model,self.conf_ema_model,
                                                           c_feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                           c_adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                           c_coords[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                           c_masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                           step_id,
                                                           FGP.final_timesteps,
                                                           fix_masks=c_fix_labels[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                           flex_masks=c_flexible_labels[i*FGP.batchsize:(i+1)*FGP.batchsize])
                if torch.isnan(predicted).any():
                    print ('NAN in predicted')
                p_predicted=predicted[:,:FGP.max_patoms]
                p_target=target[:,:FGP.max_patoms]
                p_masks=c_masks[i*FGP.batchsize:(i+1)*FGP.batchsize,:FGP.max_patoms]
                p_loss_dismat,p_loss_bonddis,p_loss_angle,p_loss_dihedral=self.IC_Loss(p_predicted,p_target,
                                                                               p_zbs[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                               p_zas[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                               p_zds[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                               p_zbmasks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                               p_zamasks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                               p_zdmasks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                               p_masks)
                
                l_predicted=predicted[:,FGP.max_patoms:]
                l_target=target[:,FGP.max_patoms:]
                l_masks=c_masks[i*FGP.batchsize:(i+1)*FGP.batchsize,FGP.max_patoms:]
                l_loss_dismat,l_loss_bonddis,l_loss_angle,l_loss_dihedral=self.IC_Loss(l_predicted,l_target,
                                                                                 l_zbs[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_zas[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_zds[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_zbmasks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_zamasks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_zdmasks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_masks)
                
                p_ic_loss=p_loss_angle+5*p_loss_bonddis+p_loss_dihedral+p_loss_dismat
                p_xyz_loss=F.mse_loss(p_predicted[p_masks.bool()],p_target[p_masks.bool()])
                
                l_ic_loss=l_loss_angle+5*l_loss_bonddis+l_loss_dihedral+l_loss_dismat
                l_xyz_loss=F.mse_loss(l_predicted[l_masks.bool()],l_target[l_masks.bool()])
                
                pl_dismat_pred=torch.cdist(p_predicted,l_predicted,compute_mode='donot_use_mm_for_euclid_dist')
                pl_dismat_target=torch.cdist(p_target,l_target,compute_mode='donot_use_mm_for_euclid_dist')
                pl_masks=(p_masks.unsqueeze(-1)*l_masks.unsqueeze(-1).permute(0,2,1)).bool()
                
                l_pl_dismat=F.mse_loss(pl_dismat_pred[pl_masks],pl_dismat_target[pl_masks])
                
                loss=l_ic_loss*FGP.loss_weight["ic"]+l_xyz_loss*FGP.loss_weight["xyz"]+l_pl_dismat
                
                loss+=(p_ic_loss*FGP.loss_weight["ic"]+p_xyz_loss*FGP.loss_weight["xyz"])*0.2
 
                if mode=='train':
                    loss.backward()
                    
                total_loss+=loss.item()
                total_loss_dismat+=l_loss_dismat.item()
                total_loss_bonddis+=l_loss_bonddis.item()
                total_loss_angle+=l_loss_angle.item()
                total_loss_dihedral+=l_loss_dihedral.item()
                total_loss_xyz+=l_xyz_loss.item()
                total_loss_pl+=l_pl_dismat.item()
                if FGP.ec_mode=='inpaint':
                    total_loss_dismat+=p_loss_dismat.item()
                    total_loss_bonddis+=p_loss_bonddis.item()
                    total_loss_angle+=p_loss_angle.item()
                    total_loss_dihedral+=p_loss_dihedral.item()
                    total_loss_xyz+=p_xyz_loss.item()
                    
        total_loss=total_loss/num
        total_loss_dismat=total_loss_dismat/num
        total_loss_bonddis=total_loss_bonddis/num
        total_loss_angle=total_loss_angle/num
        total_loss_dihedral=total_loss_dihedral/num
        total_loss_xyz=total_loss_xyz/num
        total_loss_pl=total_loss_pl/num
        
        self.lr=self.optim.state_dict()['param_groups'][0]['lr']

        lstr=f'DM: {total_loss_dismat:.3E}, B: {total_loss_bonddis:.3E}, A: {total_loss_angle:.3E}, T: {total_loss_dihedral:.3E}, R: {total_loss_xyz:.3E}, PL: {total_loss_pl:.3E}'
        if mode=='train':
            for group in self.optim.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], 1.0, 2)
        
            self.optim.step()

            num_timesteps=timesteps_schedule(step_id,FGP.final_timesteps,initial_timesteps=FGP.initial_timesteps,final_timesteps=FGP.final_timesteps)
        
            ema_decay_rate = ema_decay_rate_schedule(
                                num_timesteps,
                                initial_ema_decay_rate=0.95,
                                initial_timesteps=2,
                            )
        
            update_ema_model(self.conf_ema_model,self.conf_online_model,ema_decay_rate)
            
        return total_loss,total_loss_dismat,total_loss_bonddis,total_loss_angle,total_loss_dihedral,total_loss_xyz,lstr
    
    def Ligand_Coord_Gen_Datas_to_gpu(self,Datas):
        l_feats=self.__to_device(Datas["L_Feats"].float())
        l_atoms=self.__to_device(Datas["L_Atoms"].float())
        l_adjs_mat=self.__to_device(Datas["L_Adjs_Mat"].float())
        l_coords=self.__to_device(Datas["L_Coords"].float())
        l_masks=self.__to_device(Datas["L_Masks"])
        l_fix_labels=self.__to_device(Datas["L_Fix_Masks"].float())
        l_flexible_labels=self.__to_device(Datas["L_Flexible_Masks"].float())
        (l_zbs,l_zas,l_zds)=(self.__to_device(Datas[key]) for key in ["L_Zbs","L_Zas","L_Zds"])
        (l_zbmasks,l_zamasks,l_zdmasks)=(self.__to_device(Datas[key].bool()) for key in ["L_Zbmask","L_Zamask","L_Zdmask"])
        l_pocket_labels=self.__to_device(Datas["L_Pocket_Labels"])
        l_ligand_labels=self.__to_device(Datas["L_Ligand_Labels"])
        return l_atoms,l_feats,l_adjs_mat,l_coords,l_fix_labels,l_flexible_labels,l_masks,\
            l_zbs,l_zas,l_zds,l_zbmasks,l_zamasks,l_zdmasks,\
            l_pocket_labels,l_ligand_labels
            
    def Train_Ligand_Conf_Step(self,Datas,step_id,mode='train'):
        l_atoms,l_feats,l_adjs_mat,l_coords,l_fix_labels,l_flexible_labels,l_masks,\
            l_zbs,l_zas,l_zds,l_zbmasks,l_zamasks,l_zdmasks,\
                    l_pocket_labels,l_ligand_labels=self.Ligand_Coord_Gen_Datas_to_gpu(Datas)
        self.optim.zero_grad()

        total_loss=0
        total_loss_dismat=0
        total_loss_bonddis=0
        total_loss_angle=0
        total_loss_dihedral=0
        total_loss_xyz=0

        num=0
        for i in range(FGP.accsteps):
            num+=1
            if len(l_feats[i*FGP.batchsize:(i+1)*FGP.batchsize])>0:
                predicted,target=self.consistency_training(self.conf_online_model,self.conf_ema_model,
                                                           l_feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                           l_adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                           l_coords[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                           l_masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                           step_id,
                                                           FGP.final_timesteps,
                                                           fix_masks=l_fix_labels[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                           flex_masks=l_flexible_labels[i*FGP.batchsize:(i+1)*FGP.batchsize])
                if torch.isnan(predicted).any():
                    print ('NAN in predicted')
                #print (predicted.shape,target.shape)    
                l_loss_dismat,l_loss_bonddis,l_loss_angle,l_loss_dihedral=self.IC_Loss(predicted,target,
                                                                                 l_zbs[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_zas[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_zds[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_zbmasks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_zamasks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_zdmasks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                                                 l_masks[i*FGP.batchsize:(i+1)*FGP.batchsize])
                
                l_ic_loss=l_loss_angle+5*l_loss_bonddis+l_loss_dihedral+l_loss_dismat
                l_xyz_loss=F.mse_loss(predicted[l_masks[i*FGP.batchsize:(i+1)*FGP.batchsize].bool()],target[l_masks[i*FGP.batchsize:(i+1)*FGP.batchsize].bool()])
                
                loss=l_ic_loss*FGP.loss_weight["ic"]+l_xyz_loss*FGP.loss_weight["xyz"]
 
                if mode=='train':
                    loss.backward()
                    
                total_loss+=loss.item()
                total_loss_dismat+=l_loss_dismat.item()
                total_loss_bonddis+=l_loss_bonddis.item()
                total_loss_angle+=l_loss_angle.item()
                total_loss_dihedral+=l_loss_dihedral.item()
                total_loss_xyz+=l_xyz_loss.item()
                    
        total_loss=total_loss/num
        total_loss_dismat=total_loss_dismat/num
        total_loss_bonddis=total_loss_bonddis/num
        total_loss_angle=total_loss_angle/num
        total_loss_dihedral=total_loss_dihedral/num
        total_loss_xyz=total_loss_xyz/num
        
        self.lr=self.optim.state_dict()['param_groups'][0]['lr']

        lstr=f'DM: {total_loss_dismat:.3E}, B: {total_loss_bonddis:.3E}, A: {total_loss_angle:.3E}, T: {total_loss_dihedral:.3E}, R: {total_loss_xyz:.3E}'
        if mode=='train':
            for group in self.optim.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], 1.0, 2)
        
            self.optim.step()

            num_timesteps=timesteps_schedule(step_id,FGP.final_timesteps,initial_timesteps=FGP.initial_timesteps,final_timesteps=FGP.final_timesteps)
        
            ema_decay_rate = ema_decay_rate_schedule(
                                num_timesteps,
                                initial_ema_decay_rate=0.95,
                                initial_timesteps=2,
                            )
        
            update_ema_model(self.conf_ema_model,self.conf_online_model,ema_decay_rate)
            
        return total_loss,total_loss_dismat,total_loss_bonddis,total_loss_angle,total_loss_dihedral,total_loss_xyz,lstr
    
    def KL_Loss(self,output,target,termination_weight=1):
        """
        The graph generation loss is the KL divergence between the target and
        predicted actions.
        Args:
        ----
            output (torch.Tensor)        : Predicted APD tensor.
            target_output (torch.Tensor) : Target APD tensor.

        Returns:
        -------
            loss (torch.Tensor) : Average loss for this output.
        """
        # define activation function; note that one must use the softmax in the
        # KLDiv, never the sigmoid, as the distribution must sum to 1
        LogSoftmax = torch.nn.LogSoftmax(dim=1)
        output     = LogSoftmax(output)
        # normalize the target output (as can contain information on > 1 graph)
        target_output = target/torch.sum(target, dim=1, keepdim=True)
        # define loss function and calculate the los
        # criterion = torch.nn.KLDivLoss(reduction="batchmean")
        criterion = torch.nn.KLDivLoss(reduction="none")
        weight=torch.ones_like(target_output).cuda()
        if termination_weight!=1:
            weight[:,-1]=termination_weight
        loss = criterion(target=target_output, input=output)*weight
        #print (target_output.shape,output.shape,loss.shape)
        loss = loss.sum() / output.size(0)
        #print (loss.shape)
        return loss 
    
    def Leaf_Add_Datas_to_gpu(self,Datas,dtype='Complex'):
        if dtype=='Complex':
            feats=self.__to_device(Datas["C_Feats"].float())
            adjs_mat=self.__to_device(Datas["C_Adjs_Mat"].float())
            coords=self.__to_device(Datas["C_Coords"].float())
            masks=self.__to_device(Datas["C_Masks"])
            
        else:
            feats=self.__to_device(Datas["L_Feats"].float())
            adjs_mat=self.__to_device(Datas["L_Adjs_Mat"].float())
            coords=self.__to_device(Datas["L_Coords"].float())
            masks=self.__to_device(Datas["L_Masks"])
            
        l_apds=self.__to_device(Datas["L_Apds"].float())
        return feats,adjs_mat,coords,masks,l_apds
    
    def Train_Leaf_Add_Step(self,Datas,mode='train',dtype='Complex'):
        if mode!='train':
            self.node_add_model.eval()
        else:
            self.node_add_model.train()
        
        feats,adjs_mat,coords,masks,l_apds=self.Leaf_Add_Datas_to_gpu(Datas,dtype)
            
        self.optim.zero_grad()
        total_loss=0
        num=0
        for i in range(FGP.accsteps):
            num+=1
            if len(feats[i*FGP.batchsize:(i+1)*FGP.batchsize])>0:
                
                APD_pred=self.node_add_model(
                                                feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                coords[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                masks[i*FGP.batchsize:(i+1)*FGP.batchsize]
                                            )
                
                nadd_loss=self.KL_Loss(APD_pred,l_apds[i*FGP.batchsize:(i+1)*FGP.batchsize],termination_weight=10)

                if mode=='train':
                    nadd_loss.backward()
                total_loss+=nadd_loss.item()

        total_loss=total_loss/num
        
        self.lr=self.optim.state_dict()['param_groups'][0]['lr']
        
        lstr=f'Nadd: {total_loss:.3E}'
        
        if mode=='train':
            for group in self.optim.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], 1.0, 2)
        
            self.optim.step()
            
        return total_loss,lstr

    def Leaf_Rgen_Datas_to_gpu(self,Datas,dtype='Complex'):
        if dtype=='Complex':
            feats=self.__to_device(Datas["C_Feats"].float())
            adjs_mat=self.__to_device(Datas["C_Adjs_Mat"].float())
            coords=self.__to_device(Datas["C_Coords"].float())
            masks=self.__to_device(Datas["C_Masks"])
        else:
            feats=self.__to_device(Datas["L_Feats"].float())
            adjs_mat=self.__to_device(Datas["L_Adjs_Mat"].float())
            coords=self.__to_device(Datas["L_Coords"].float())
            masks=self.__to_device(Datas["L_Masks"])
            
        r_feats=self.__to_device(Datas["R_Feats"].float())
        r_adjs_mat=self.__to_device(Datas["R_Adjs_Mat"].float())
        r_masks=self.__to_device(Datas["R_Masks"])
        r_ftypes=self.__to_device(Datas["R_Ftypes"].float())
        r_apds=self.__to_device(Datas["R_Apds"].float())

        return feats,adjs_mat,coords,masks,r_feats,r_adjs_mat,r_masks,r_ftypes,r_apds

    def Train_Ring_Gen_Step(self,Datas,mode='train',dtype='Complex'):
        if mode!='train':
            self.ring_gen_model.eval()
        else:
            self.ring_gen_model.train()

        feats,adjs_mat,coords,masks,r_feats,r_adjs_mat,r_masks,r_ftypes,l_apds=self.Leaf_Rgen_Datas_to_gpu(Datas,dtype=dtype)

        self.optim.zero_grad()

        total_loss=0
        num=0
        for i in range(FGP.accsteps):
            num+=1
            if len(feats[i*FGP.batchsize:(i+1)*FGP.batchsize])>0:
                APD_pred=self.ring_gen_model(
                                                feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                coords[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                r_feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                r_adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                r_ftypes[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                r_masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                            )
                
                rgen_loss=self.KL_Loss(APD_pred,l_apds[i*FGP.batchsize:(i+1)*FGP.batchsize],termination_weight=10)
                
                if mode=='train':
                    rgen_loss.backward()
                total_loss+=rgen_loss.item()

        total_loss=total_loss/num
        
        self.lr=self.optim.state_dict()['param_groups'][0]['lr']
        
        lstr=f'Rgen: {total_loss:.3E}'
        
        if mode=='train':
            for group in self.optim.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], 1.5, 2)
            
            self.optim.step()
            
        return total_loss,lstr 
    
    def Leaf_Conn_Datas_to_gpu(self,Datas,dtype='Complex'):
        if dtype=='Complex':
            feats=self.__to_device(Datas["C_Feats"].float())
            adjs_mat=self.__to_device(Datas["C_Adjs_Mat"].float())
            coords=self.__to_device(Datas["C_Coords"].float())
            masks=self.__to_device(Datas["C_Masks"])
        else:
            feats=self.__to_device(Datas["L_Feats"].float())
            adjs_mat=self.__to_device(Datas["L_Adjs_Mat"].float())
            coords=self.__to_device(Datas["L_Coords"].float())
            masks=self.__to_device(Datas["L_Masks"])
            
        r_feats=self.__to_device(Datas["R_Feats"].float())
        r_adjs_mat=self.__to_device(Datas["R_Adjs_Mat"].float())
        r_masks=self.__to_device(Datas["R_Masks"])
        focus_atom=self.__to_device(Datas["Focus_Atom"].float())
        l_apds=self.__to_device(Datas["L_Apds"].float())
        return feats,adjs_mat,coords,masks,r_feats,r_adjs_mat,r_masks,focus_atom,l_apds
    
    def Train_Node_Conn_Step(self,Datas,mode='train',dtype='Complex'):
        if mode!='train':
            self.node_conn_model.eval()
        else:
            self.node_conn_model.train()
            
        feats,adjs_mat,coords,masks,r_feats,r_adjs_mat,r_masks,focus_atom,l_apds=self.Leaf_Conn_Datas_to_gpu(Datas,dtype=dtype)
        self.optim.zero_grad()
        total_loss=0
        num=0
        for i in range(FGP.accsteps):
            num+=1
            if len(feats[i*FGP.batchsize:(i+1)*FGP.batchsize])>0:
                APD_pred=self.node_conn_model(
                                                feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                coords[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                r_feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                r_adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                focus_atom[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                r_masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                            )
                #print ('NCONN APDs',l_apds.shape,len(l_apds.shape))
                nconn_loss=self.KL_Loss(APD_pred,l_apds[i*FGP.batchsize:(i+1)*FGP.batchsize])

                if mode=='train':
                    nconn_loss.backward()
                total_loss+=nconn_loss.item()

        total_loss=total_loss/num
        
        self.lr=self.optim.state_dict()['param_groups'][0]['lr']
        
        lstr=f'NConn: {total_loss:.3E}'
        
        if mode=='train':
            for group in self.optim.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], 1.5, 2)
        
            self.optim.step()
        return total_loss,lstr  

    def Leaf_Int_Datas_to_gpu(self,Datas):
        cg_feats=self.__to_device(Datas["CG_Feats"].float())
        cg_adjs_mat=self.__to_device(Datas["CG_Adjs_Mat"].float())
        cg_masks=self.__to_device(Datas["CG_Masks"])
        cd_feats=self.__to_device(Datas["CD_Feats"].float())
        cd_adjs_mat=self.__to_device(Datas["CD_Adjs_Mat"].float())
        cd_coords=self.__to_device(Datas["CD_Coords"].float())
        cd_masks=self.__to_device(Datas["CD_Masks"])
        focus_lgroups=self.__to_device(Datas["Focus_Lgroups"].long())
        focus_lgroups_mask=self.__to_device(Datas["Focus_Lgroups_Masks"])
        pgroups=self.__to_device(Datas["P_Groups"].long())
        pgroups_Masks=self.__to_device(Datas["P_Groups_Masks"])
        pgroups_int_masks=self.__to_device(Datas["P_INT_Groups_Masks"])
        focus_ftypes=self.__to_device(Datas["Ftypes"].float())
        l_apds=self.__to_device(Datas["L_Apds"].float())
        return cg_feats,cg_adjs_mat,cg_masks,cd_feats,cd_adjs_mat,cd_coords,cd_masks,focus_lgroups,focus_lgroups_mask,pgroups,pgroups_Masks,pgroups_int_masks,focus_ftypes,l_apds

    def Train_Node_Int_Step(self,Datas,mode='train'):
        if mode!='train':
            self.node_int_model.eval()
        else:
            self.node_int_model.train()
            
        cg_feats,cg_adjs_mat,cg_masks,cd_feats,cd_adjs_mat,cd_coords,cd_masks,focus_lgroups,focus_lgroups_mask,pgroups,pgroups_Masks,pgroups_int_masks,focus_ftypes,l_apds=self.Leaf_Int_Datas_to_gpu(Datas)
        self.optim.zero_grad()
        total_loss=0
        num=0
        for i in range(FGP.accsteps):
            num+=1
            if len(cg_feats[i*FGP.batchsize:(i+1)*FGP.batchsize])>0:
                APD_pred=self.node_int_model(
                                                cg_feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                cg_adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                cg_masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                cd_feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                cd_adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                cd_coords[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                cd_masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                pgroups[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                pgroups_Masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                pgroups_int_masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                focus_lgroups[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                focus_lgroups_mask[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                focus_ftypes[i*FGP.batchsize:(i+1)*FGP.batchsize]
                                            )
                
                nint_loss=self.KL_Loss(APD_pred,l_apds[i*FGP.batchsize:(i+1)*FGP.batchsize],termination_weight=10)

                if mode=='train':
                    nint_loss.backward()
                total_loss+=nint_loss.item()

        total_loss=total_loss/num
        
        self.lr=self.optim.state_dict()['param_groups'][0]['lr']
        
        lstr=f'Nint: {total_loss:.3E}'
        
        if mode=='train':
            for group in self.optim.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], 1.5, 2)
        
            self.optim.step()
        return total_loss,lstr 

    def Fit(self,MGFiles,Epochs=100,split_rate=0.95,mode='coords',nfiles_per_mini_epochs=50,dtype='Complex'):
        if dtype!='Complex':
            FGP.max_patoms=0

        if mode=='coords':
            self.optim=Adam(self.conf_online_model.parameters(), lr = FGP.initlr, betas=(0.5,0.999))
        elif mode=='nadd':
                self.optim=Adam(filter(lambda p:p.requires_grad, self.node_add_model.parameters()), lr = FGP.initlr, betas=(0.5,0.999))
        elif mode=='rgen':
            self.optim=Adam(filter(lambda p:p.requires_grad, self.ring_gen_model.parameters()), lr = FGP.initlr, betas=(0.5,0.999))
        elif mode=='nconn':
            self.optim=Adam(filter(lambda p:p.requires_grad, self.node_conn_model.parameters()), lr = FGP.initlr, betas=(0.5,0.999))
        elif mode=='nint':
            self.optim=Adam(filter(lambda p:p.requires_grad, self.node_int_model.parameters()), lr = FGP.initlr, betas=(0.5,0.999))

        self.lr_scheduler= ReduceLROnPlateau(
                self.optim, mode='min',
                factor=0.9, patience=FGP.lr_patience,
                verbose=True, threshold=0.0001, threshold_mode='rel',
                cooldown=FGP.lr_cooldown,
                min_lr=1e-06, eps=1e-06)

        n_mini_epochs=math.ceil(len(MGFiles)/nfiles_per_mini_epochs)
        
        num=0
        for epoch in range(Epochs):
            for mini_epoch in range(n_mini_epochs):
                
                MGs=[]
                for Fname in MGFiles[mini_epoch*nfiles_per_mini_epochs:(mini_epoch+1)*nfiles_per_mini_epochs]:
                    with open(Fname,'rb') as f:
                        comp=pickle.load(f)
                        MGs.append(comp)

                cutnum=math.ceil(len(MGs)*split_rate)
                #cutnum=0 
                if self.local_rank is not None:
                    if dtype=='Complex':
                        Train_Dataset=MG_Dataset(MGs[:cutnum],name='trainset',mode=mode)
                    else:
                        Train_Dataset=Lig_Dataset(MGs[:cutnum],name='trainset',mode=mode)
                        
                    train_sampler=torch.utils.data.distributed.DistributedSampler(Train_Dataset)
                    trainloader=DataLoader(Train_Dataset,batch_size=self.batchsize*FGP.accsteps,shuffle=False,num_workers=FGP.n_workers,sampler=train_sampler)
                    train_sampler.set_epoch(epoch)
                    if dtype=='Complex':
                        Valid_Dataset=MG_Dataset(MGs[cutnum:],name='validset',mode=mode)
                    else:
                        Valid_Dataset=Lig_Dataset(MGs[cutnum:],name='validset',mode=mode)
                    valid_sampler=torch.utils.data.distributed.DistributedSampler(Valid_Dataset)
                    validloader=DataLoader(Valid_Dataset,batch_size=self.batchsize*FGP.accsteps,shuffle=False,num_workers=FGP.n_workers,sampler=valid_sampler)
                    valid_sampler.set_epoch(epoch)
                    
                else:
                    if dtype=='Complex':
                        Train_Dataset=MG_Dataset(MGs[:cutnum],name='trainset',mode=mode)
                    else:
                        Train_Dataset=Lig_Dataset(MGs[:cutnum],name='trainset',mode=mode)
                    trainloader=DataLoader(Train_Dataset,batch_size=self.batchsize*FGP.accsteps,shuffle=False,num_workers=FGP.n_workers)
                    if dtype=='Complex':
                        Valid_Dataset=MG_Dataset(MGs[cutnum:],name='validset',mode=mode)
                    else:
                        Valid_Dataset=Lig_Dataset(MGs[cutnum:],name='validset',mode=mode)
                    validloader=DataLoader(Valid_Dataset,batch_size=self.batchsize*FGP.accsteps,shuffle=False,num_workers=FGP.n_workers)
                
                trainbar=enumerate(trainloader)
                validbar=enumerate(validloader)
                self.train_epoch_loss=0

                ntrain_batchs=math.ceil(cutnum/self.batchsize)
                nvalid_batchs=math.ceil((len(MGs)-cutnum)/self.batchsize)

                for bid,Datas in trainbar:
                    train_batch_loss=0
                    if mode=='coords':
                        steps_schedule=[sid for sid in range(25)]+[50,100,150]
                        for step in steps_schedule:
                            if dtype=='Complex':
                                step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_loss_xyz,step_lstr=self.Train_Complex_Conf_Step(Datas,step_id=step,mode='train')
                            else:
                                step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_loss_xyz,step_lstr=self.Train_Ligand_Conf_Step(Datas,step_id=step,mode='train')
                            lstr=f'Training -- Epochs: {epoch},{mini_epoch} bid: {bid} step: {step} lr: {self.lr:.3E} '+step_lstr
                            print (lstr)
                            self.logger_dict['coords'].write(lstr+'\n')
                            self.logger_dict['coords'].flush()
                            train_batch_loss+=step_loss
                            
                        self.train_batch_loss=train_batch_loss
                        self.lr_scheduler.step(metrics=self.train_batch_loss)
                        self.train_epoch_loss+=train_batch_loss
                        torch.cuda.empty_cache() 
                        if self.local_rank is not None:
                            dist.barrier()
                        
                    elif mode=='nadd':
                        step_loss,step_lstr=self.Train_Leaf_Add_Step(Datas,mode='train',dtype=dtype)
                        lstr=f'Training -- Epochs: {epoch},{mini_epoch} bid: {bid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_dict['nadd'].write(lstr+'\n')
                        self.logger_dict['nadd'].flush()
                        self.train_batch_loss=step_loss
                        self.lr_scheduler.step(metrics=self.train_batch_loss)
                        torch.cuda.empty_cache()
                        
                    elif mode=='rgen':
                        step_loss,step_lstr=self.Train_Ring_Gen_Step(Datas,mode='train',dtype=dtype)
                        lstr=f'Training -- Epochs: {epoch},{mini_epoch} bid: {bid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_dict['rgen'].write(lstr+'\n')
                        self.logger_dict['rgen'].flush()
                        self.train_batch_loss=step_loss
                        self.lr_scheduler.step(metrics=self.train_batch_loss)
                        torch.cuda.empty_cache()
                        
                    elif mode=='nconn':
                        step_loss,step_lstr=self.Train_Node_Conn_Step(Datas,mode='train',dtype=dtype)
                        lstr=f'Training -- Epochs: {epoch},{mini_epoch} bid: {bid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_dict['nconn'].write(lstr+'\n')
                        self.logger_dict['nconn'].flush()
                        self.train_batch_loss=step_loss
                        self.lr_scheduler.step(metrics=self.train_batch_loss)
                        torch.cuda.empty_cache()
                        
                    elif mode=='nint':
                        assert dtype=='Complex', 'Only Complex is supported for Node Interaction'
                        step_loss,step_lstr=self.Train_Node_Int_Step(Datas,mode='train')
                        lstr=f'Training -- Epochs: {epoch},{mini_epoch} bid: {bid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_dict['nint'].write(lstr+'\n')
                        self.logger_dict['nint'].flush()
                        self.train_batch_loss=step_loss
                        self.lr_scheduler.step(metrics=self.train_batch_loss)
                        torch.cuda.empty_cache()
                 
                
                self.valid_epoch_loss=0

                for vid,vDatas in validbar:
                    valid_batch_loss=0
                    if mode=='coords':
                        for step in range(FGP.final_timesteps):
                            with torch.no_grad():
                                if dtype=='Complex':
                                    step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_loss_xyz,step_lstr=self.Train_Complex_Conf_Step(vDatas,step_id=step,mode='eval')
                                else:
                                    step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_loss_xyz,step_lstr=self.Train_Ligand_Conf_Step(vDatas,step_id=step,mode='eval')
                            lstr=f'Valid    -- Epochs: {epoch},{mini_epoch} bid: {vid} step: {step} lr: {self.lr:.3E} '+step_lstr
                            print (lstr+'\n')
                            self.logger_dict['coords'].write(lstr+'\n')
                            self.logger_dict['coords'].flush()
                            valid_batch_loss+=step_loss 
                        self.valid_epoch_loss+=valid_batch_loss
                        self.logger_dict['coords'].write(f'{Fname} Valid    -- Epochs: validloss: {self.valid_epoch_loss/nvalid_batchs:.3E}') 
                        self.logger_dict['coords'].flush()
                        torch.cuda.empty_cache() 
                        if self.local_rank is not None:
                            dist.barrier()
                    if mode=='nadd':
                        with torch.no_grad():
                            step_loss,step_lstr=self.Train_Leaf_Add_Step(vDatas,mode='eval',dtype=dtype)
                        lstr=f'Valid    -- Epochs: {epoch},{mini_epoch} bid: {vid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_dict['nadd'].write(lstr+'\n')
                        self.logger_dict['nadd'].flush()
                        valid_batch_loss+=step_loss
                        self.valid_epoch_loss+=valid_batch_loss
                        torch.cuda.empty_cache()
                    elif mode=='rgen':
                        with torch.no_grad():
                            step_loss,step_lstr=self.Train_Ring_Gen_Step(vDatas,mode='eval',dtype=dtype)
                        lstr=f'Valid    -- Epochs: {epoch},{mini_epoch} bid: {vid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_dict['rgen'].write(lstr+'\n')
                        self.logger_dict['rgen'].flush()
                        valid_batch_loss+=step_loss
                        self.valid_epoch_loss+=valid_batch_loss
                        torch.cuda.empty_cache()
                    elif mode=='nconn':
                        with torch.no_grad():
                            step_loss,step_lstr=self.Train_Node_Conn_Step(vDatas,mode='eval',dtype=dtype)
                        lstr=f'Valid    -- Epochs: {epoch},{mini_epoch} bid: {vid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_dict['nconn'].write(lstr+'\n')
                        self.logger_dict['nconn'].flush()
                        valid_batch_loss+=step_loss
                        self.valid_epoch_loss+=valid_batch_loss
                        torch.cuda.empty_cache()
                    elif mode=='nint':
                        with torch.no_grad():
                            step_loss,step_lstr=self.Train_Node_Int_Step(vDatas,mode='eval')
                        lstr=f'Valid    -- Epochs: {epoch},{mini_epoch} bid: {vid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_dict['nint'].write(lstr+'\n')
                        self.logger_dict['nint'].flush()
                        valid_batch_loss+=step_loss
                        self.valid_epoch_loss+=valid_batch_loss
                        torch.cuda.empty_cache()
                print (self.valid_epoch_loss,self.min_valid_loss_epoch)
                if self.valid_epoch_loss<self.min_valid_loss_epoch:
                    self.min_valid_loss_epoch=self.valid_epoch_loss
                    print (f'Save New check point of model at Epoch:{epoch},{mini_epoch}')
                    if self.local_rank==0 or self.local_rank is None:
                        self.save_cpkt(mode='minloss')

                if self.local_rank==0 or self.local_rank is None:
                    self.save_cpkt(mode='perepoch')
                num+=1
                if self.local_rank is not None:
                    dist.barrier()
        return 
    
    def sample_coords_batch(self,g_feats,g_atoms,g_adjs_mat,g_coords,g_masks,g_pocket_labels=None,g_ligand_labels=None,
                            with_MMFF_guide=False,guide_loops=1,guide_type='asynchronous',show_state=True,guide_model="LBFGS",rdkit_mols=None):
        sigmas = karras_schedule(
            FGP.final_timesteps, FGP.sigma_min, FGP.sigma_max, FGP.rho, g_coords.device
        )
        sigmas= self.__to_device(reversed(sigmas)[:-1])
        if g_pocket_labels is None:
            g_pocket_labels=self.__to_device(torch.zeros_like(g_ligand_labels))
        
        samples,sample_diff, sample_diff_before, energies= self.consistency_sampling_and_editing(
                                self.conf_online_model,
                                atoms=g_atoms,
                                feats=g_feats,
                                adjs=g_adjs_mat,
                                ligand_labels=g_ligand_labels,
                                pocket_labels=g_pocket_labels,
                                y=torch.randn_like(g_coords).to(g_coords), # used to infer the shapes
                                gmasks=g_masks,
                                sigmas=sigmas, # sampling starts at the maximum std (T)
                                clip_denoised=False, # whether to clamp values to [-1, 1] range
                                verbose=True,
                                with_MMFF_guide=with_MMFF_guide,
                                guide_loops=guide_loops,
                                guide_type=guide_type,
                                show_state=True,
                                min_type=guide_model,
                                rdkit_mols=rdkit_mols,
                            )
        
        return samples ,sample_diff, sample_diff_before, energies
    
    def inpaint_coords_batch(self,c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,c_fix_labels,c_flexible_labels,c_pocket_labels,c_ligand_labels,
                             with_MMFF_guide=False,guide_loops=1,guide_type='asynchronous',show_state=True,guide_model="LBFGS",rdkit_mols=None,pocket_mols=None,ligand_mols=None):
        # coords generator fails in 3 aspect, 
        # 1. the pocket_labels should be provided, when ligand labels provided, the mass center is incorrect for docking model.
        # 2. the c_flexible_masks should be zeros in ghost padding positions
        # 3. the pl_int_adjs should be atom-based interaction adjs instead of group-divided atom adjs. 
        #c_atoms,c_feats,c_adjs_mat,c_coords,c_fix_labels,c_flexible_labels,c_masks,c_zbs,c_zas,c_zds,c_zbmasks,c_zamasks,c_zdmasks,c_pocket_labels,c_ligand_labels=self.Coord_Gen_Datas_to_gpu(Datas)
        
        inpaint_coords=c_coords*c_fix_labels.unsqueeze(-1).float()
        #print (c_flexible_labels)
        sigmas = karras_schedule(
            FGP.final_timesteps, FGP.sigma_min, FGP.sigma_max, FGP.rho, c_coords.device
        )
        sigmas= self.__to_device(reversed(sigmas)[:-1]).float()
        samples,samples_diff,sample_diff_before,energies = self.consistency_sampling_and_editing(
                                self.conf_online_model,
                                atoms=c_atoms,
                                feats=c_feats,
                                adjs=c_adjs_mat,
                                ligand_labels=c_ligand_labels,
                                pocket_labels=c_pocket_labels,
                                y=inpaint_coords, # used to infer the shapes
                                gmasks=c_masks,
                                mask=c_flexible_labels.unsqueeze(-1).float(),
                                sigmas=sigmas, # sampling starts at the maximum std (T)
                                clip_denoised=False, # whether to clamp values to [-1, 1] range
                                verbose=True,
                                with_MMFF_guide=with_MMFF_guide,
                                guide_loops=guide_loops,
                                guide_type=guide_type,
                                show_state=True,
                                min_type=guide_model,
                                rdkit_mols=rdkit_mols,
                                pocket_mols=pocket_mols,
                                ligand_mols=ligand_mols
                            )

        return samples,samples_diff,sample_diff_before, energies 
    
    def Check_Conf_Model(self,MG,conf_num_per_states=10,batchsize=1,savepath='./sample',mode='complex'):

        Dataset=MG_Dataset([MG],name='sample',mode='coords')

        Dataset.repulicate(rep_num=conf_num_per_states)

        loader=DataLoader(Dataset,batch_size=batchsize,shuffle=False,num_workers=FGP.n_workers)
        bar=enumerate(loader)
        total_samples=[]
        total_samples_diff=[]
        total_samples_diff_before=[]
        pocket_mol=MG.Trans_Pocket_to_Mol()
        Chem.SanitizeMol(pocket_mol)
        for bid,Datas in bar:
            with torch.no_grad():
                c_atoms,c_feats,c_adjs_mat,c_coords,c_fix_labels,c_flexible_labels,c_masks,\
                    l_zbs,l_zas,l_zds,l_zbmasks,l_zamasks,l_zdmasks,\
                    p_zbs,p_zas,p_zds,p_zbmasks,p_zamasks,p_zdmasks,\
                    c_pocket_labels,c_ligand_labels,step_ids=self.Complex_Coord_Gen_Datas_to_gpu(Datas) 
                
                c_mols=[]
                for sid in step_ids:
                    l_gmasks=MG.crd_gen_states[0][sid]
                    l_mol=MG.Trans_Ligand_to_Mol(lmasks=l_gmasks)
                    Chem.SanitizeMol(l_mol)
                    c_mol= Chem.CombineMols(pocket_mol, l_mol)
                    print (sid,Chem.MolToSmiles(l_mol))
                    c_mols.append(c_mol)
                
                samples,samples_diff,samples_diff_before,_=self.inpaint_coords_batch(\
                                    c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,\
                                    c_fix_labels,c_flexible_labels,c_pocket_labels,c_ligand_labels,
                                    with_MMFF_guide=FGP.with_MMFF_guide,
                                    guide_loops=FGP.MMFF_guide_loops,
                                    guide_type=FGP.MMFF_guide_type,
                                    show_state=True,
                                    guide_model=FGP.MMFF_guide_model,
                                    rdkit_mols=c_mols)
            
            samples_diff=torch.concat(samples_diff,axis=1)
            samples_diff_before=torch.concat(samples_diff_before,axis=1)
            #print (samples_diff.shape,samples_diff_before.shape)    
            total_samples.append(samples)
            total_samples_diff.append(samples_diff)
            total_samples_diff_before.append(samples_diff_before) 
        
        total_samples=torch.concat(total_samples,axis=0).clone().detach().cpu().numpy()
        total_samples_diff=torch.concat(total_samples_diff,axis=0).clone().detach().cpu().numpy()
        total_samples_diff_before=torch.concat(total_samples_diff_before,axis=0).clone().detach().cpu().numpy()

        for idx,states in enumerate(Dataset.coord_states):
            gid,stepid=states
            confid=idx-stepid*conf_num_per_states
            MG_cp=copy.deepcopy(Dataset.mglist[gid])
            lmasks=MG_cp.crd_gen_states[0][stepid]

            Mass_center=MG_cp.get_pocket_mass_center()
            MG_cp.p_coords=MG_cp.p_coords-Mass_center
            MG_cp.l_coords=MG_cp.l_coords-Mass_center
            
            ref_ligmol=MG_cp.Trans_Ligand_to_Mol(lmasks=lmasks)
            ref_pocket=MG_cp.Trans_Pocket_to_Mol()
            
            subpath=f'{gid}-{stepid}'
            os.system(f'mkdir -p {savepath}/{subpath}')
            molsupp=Chem.SDWriter(f'{savepath}/{subpath}/ref_lig-{confid}.sdf')
            molsupp.write(ref_ligmol)
            molsupp.close()
            MolToXYZ(ref_ligmol,f'{savepath}/{subpath}/ref_lig-{confid}.xyz')
            MolToXYZ(ref_pocket,f'{savepath}/{subpath}/ref_pocket-{confid}.xyz')
            p_coords=total_samples[idx][:MG_cp.p_natoms]
            l_natoms=int(np.sum(lmasks))

            l_coords=total_samples[idx][FGP.max_patoms:FGP.max_patoms+l_natoms]
            gen_ligmol=copy.deepcopy(ref_ligmol)
            gen_pocket=copy.deepcopy(ref_pocket)
            gen_ligmol=Change_mol_xyz(gen_ligmol,l_coords)
            gen_pocket=Change_mol_xyz(gen_pocket,p_coords)
            molsupp=Chem.SDWriter(f'{savepath}/{subpath}/gen_lig-{confid}.sdf')
            molsupp.write(gen_ligmol)
            molsupp.close()
            #MolToXYZ(gen_ligmol,f'{savepath}/{subpath}/gen_lig-{confid}.xyz')
            MolToXYZ(gen_pocket,f'{savepath}/{subpath}/gen_pocket-{confid}.xyz')
            if FGP.save_diff_process:
                for i in range(25):
                    os.system(f'mkdir -p {savepath}/{subpath}/{confid}_diff-{i}')
                    os.system(f'mkdir -p {savepath}/{subpath}/{confid}_diff_before-{i}')
                    p_coords=total_samples_diff[idx][i][:MG_cp.p_natoms]
                    l_natoms=int(np.sum(lmasks))
                    l_coords=total_samples_diff[idx][i][FGP.max_patoms:FGP.max_patoms+l_natoms]
                    gen_ligmol=copy.deepcopy(ref_ligmol)
                    gen_pocket=copy.deepcopy(ref_pocket)
                    gen_ligmol=Change_mol_xyz(gen_ligmol,l_coords)
                    gen_pocket=Change_mol_xyz(gen_pocket,p_coords)
                    molsupp=Chem.SDWriter(f'{savepath}/{subpath}/{confid}_diff-{i}/gen_lig-{confid}_diff-{i}.sdf')
                    molsupp.write(gen_ligmol)
                    molsupp.close()
                    MolToXYZ(gen_pocket,f'{savepath}/{subpath}/{confid}_diff-{i}/gen_pocket-{confid}_diff-{i}.xyz')
                    p_coords=total_samples_diff_before[idx][i][:MG_cp.p_natoms]
                    l_natoms=int(np.sum(lmasks))
                    l_coords=total_samples_diff_before[idx][i][FGP.max_patoms:FGP.max_patoms+l_natoms]
                    gen_ligmol=copy.deepcopy(ref_ligmol)
                    gen_pocket=copy.deepcopy(ref_pocket)
                    gen_ligmol=Change_mol_xyz(gen_ligmol,l_coords)
                    gen_pocket=Change_mol_xyz(gen_pocket,p_coords)
                    molsupp=Chem.SDWriter(f'{savepath}/{subpath}/{confid}_diff_before-{i}/gen_lig-{confid}_diff_before-{i}.sdf')
                    molsupp.write(gen_ligmol)
                    molsupp.close()
                    MolToXYZ(gen_pocket,f'{savepath}/{subpath}/{confid}_diff_before-{i}/gen_pocket-{confid}_diff_before-{i}.xyz')         
        return 
    
    def sample_actions(self,apds,temp=1.0):
        apds=torch.exp(torch.log(apds)/temp)
        action_probs=torch.distributions.Multinomial(1,probs=apds)
        action=action_probs.sample()
        return action

    def compute_likelihoods(self,apds,action):
        likelihoods=torch.log(apds[action==1])
        return likelihoods

    def pred_nadd_apd(self,gnodes,gedges,gcoords,gmasks,nadd_constrain_masks=None):
        softmax=torch.nn.Softmax(dim=1)
        nadd_pred=self.node_add_model(gnodes,gedges,gcoords,gmasks)
        if FGP.with_term_model:
            nadd_pred=nadd_pred[:,:-1]
            
        if nadd_constrain_masks is None:
            nadd_constrain_masks=torch.ones_like(nadd_pred)
        #nadd_constrain_masks[:,-1]=0 # termination will be predicted by graph_term_model
        nadd_pred=softmax(nadd_pred)*nadd_constrain_masks.to(nadd_pred)
        return nadd_pred
    
    def split_nadd_action(self,actions):
        f_nadd=actions[:,:-1]
        add_idc=torch.nonzero(f_nadd,as_tuple=True)
        if FGP.with_term_model:
            return add_idc
        else:
            f_term=actions[:,-1]
            term_idc=torch.nonzero(f_term,as_tuple=True)
            return add_idc,term_idc
     
    def nadd_step(self,g_nodes,g_edges,g_coords,g_masks,nadd_constrain_masks=None,temp=1.0):
        nadd_apd=self.pred_nadd_apd(g_nodes,g_edges,g_coords,g_masks,nadd_constrain_masks)
        nadd_action=self.sample_actions(nadd_apd,temp)
        if FGP.with_term_model:
            add_idc=self.split_nadd_action(nadd_action)
            likelihoods=self.compute_likelihoods(nadd_apd,nadd_action)
            return add_idc,likelihoods
        else:
            add_idc,term_idc=self.split_nadd_action(nadd_action)
            likelihoods=self.compute_likelihoods(nadd_apd,nadd_action)
            return add_idc,term_idc,likelihoods
    
    def pred_term_apd(self,gnodes,gedges,gcoords,gmasks,term_constrain_masks=None):
        softmax=torch.nn.Softmax(dim=1)
        term_pred=self.graph_term_model(gnodes,gedges,gcoords,gmasks)  
        term_pred=softmax(term_pred)
        #print (torch.sum(term_pred[:,:-1],dim=-1).shape)
        term_pred=torch.cat((torch.sum(term_pred[:,:-1],dim=-1,keepdim=True),term_pred[:,-1:]),dim=-1)
        print ('term_pred',term_pred.shape)
        if term_constrain_masks is None:
            term_constrain_masks=torch.ones_like(term_pred)
        #nadd_constrain_masks[:,-1]=0 # termination will be predicted by graph_term_model
        term_pred=term_pred*term_constrain_masks.to(term_pred)
        return term_pred
    
    def split_term_action(self,actions):
        f_non_term=actions[:,0]
        f_term=actions[:,-1]
        non_term_idc=torch.nonzero(f_non_term,as_tuple=True)
        term_idc=torch.nonzero(f_term,as_tuple=True)
        return non_term_idc,term_idc
    
    def term_step(self,g_nodes,g_edges,g_coords,g_masks,term_constrain_masks=None):
        term_apd=self.pred_term_apd(g_nodes,g_edges,g_coords,g_masks,term_constrain_masks)
        term_action=self.sample_actions(term_apd)
        non_term_idc,term_idc=self.split_term_action(term_action)
        likelihoods=self.compute_likelihoods(term_apd,term_action)
        return non_term_idc,term_idc,likelihoods
         
    def pred_rgen_apd(self,g_nodes,g_edges,g_coords,g_masks,r_nodes,r_edges,f_t,r_masks,rgen_constrain_masks=None):
        softmax=torch.nn.Softmax(dim=1)
        rgen_pred=self.ring_gen_model(g_nodes,g_edges,g_coords,g_masks,r_nodes,r_edges,f_t,r_masks)
        if rgen_constrain_masks is None:
            rgen_constrain_masks=torch.ones_like(rgen_pred)
        rgen_pred=softmax(rgen_pred)*rgen_constrain_masks.to(rgen_pred) 
        return rgen_pred

    def split_rgen_action(self,action):
        add_shape=(action.shape[0],*FGP.r_add_dim)
        conn_shape=(action.shape[0],*FGP.r_conn_dim)
        f_add=action[:,:np.prod(FGP.r_add_dim)].view(add_shape)
        f_conn=action[:,np.prod(FGP.r_add_dim):-1].view(conn_shape)
        f_term=action[:,-1]
        add_idc=torch.nonzero(f_add,as_tuple=True)
        conn_idc=torch.nonzero(f_conn,as_tuple=True)
        term_idc=torch.nonzero(f_term,as_tuple=True)
        return add_idc,conn_idc,term_idc

    def rgen_step(self,g_nodes,g_edges,g_coords,g_masks,r_nodes,r_edges,f_t,r_masks,r_atomnums,rgen_constrain_masks=None,temp=1.0):
        rgen_apd=self.pred_rgen_apd(g_nodes,g_edges,g_coords,g_masks,r_nodes,r_edges,f_t,r_masks,rgen_constrain_masks)
        rgen_action=self.sample_actions(rgen_apd,temp)
        likelihoods=self.compute_likelihoods(rgen_apd,rgen_action)
        add_idc,conn_idc,term_idc=self.split_rgen_action(rgen_action)
        # add bond_from to add_ids to judge the new added atom have beyond the maximum atoms
        r_add_froms=r_atomnums[add_idc[0]]
        add_idc=(*add_idc,r_add_froms)
        # add bond_from to add_ids to judge the new added atom have beyond the maximum atoms
        r_conn_froms=r_atomnums[conn_idc[0]]-1
        conn_idc=(*conn_idc,r_conn_froms)
        return add_idc,conn_idc,term_idc,likelihoods
    
    def invalid_rgen_actions(self,add_idc,conn_idc,term_idc,r_edges,r_atomnums,r_maxatoms):
        
        #ring_maxatoms=torch.where(ring_atomnum>rw_ring_maxatoms,ring_atomnum,ft_ring_maxatoms).long()         
        # get invalid indices for when adding a new node to a non-empty graph
        empty_graphs=torch.nonzero(r_atomnums[add_idc[0]]==0) # add atom connections to empty graphs
        invalid_add_to_non_empty=torch.nonzero(add_idc[1]>=r_atomnums[add_idc[0]]) # add atom connections to unexist atoms
        combined=torch.cat((invalid_add_to_non_empty,empty_graphs),dim=0).squeeze(1)
        uniques,counts=combined.unique(return_counts=True)
        invalid_add_to_non_empty_ids=uniques[counts==1].unsqueeze(dim=1) 
        
        # get invalid indices for when adding a new node to an empty graph
        invalid_add_to_empty = torch.nonzero(add_idc[1]!=r_atomnums[add_idc[0]])
        combined=torch.cat((invalid_add_to_empty,empty_graphs),dim=0).squeeze(1)
        uniques,counts=combined.unique(return_counts=True)
        invalid_add_to_empty_ids=uniques[counts>1].unsqueeze(dim=1)
        
        # get invalid indices for when connecting a node to an unpossible node which out of max ring size
        invalid_add_to_impossible_ids=torch.nonzero(add_idc[5]>=r_maxatoms[add_idc[0]])
        # get invalid indices for when "connecting" a node in a graph with zero nodes
        invalid_conn_to_empty_ids=torch.nonzero(r_atomnums[conn_idc[0]]==0)
        # get invalid indices for when connecting a node to nonexisting node
        invalid_conn_ids=torch.nonzero(conn_idc[1]>=r_atomnums[conn_idc[0]])
        # get invalid indices for when connecting a node to itself
        invalid_conn_self_ids=torch.nonzero(conn_idc[1]==conn_idc[3])
        # get invalid indices for when attemting to add multiple edges
        invalid_conn_multi_ids=torch.nonzero(torch.sum(r_edges,dim=-1)[conn_idc[0].long(),conn_idc[1].long(),conn_idc[-1].long()]==1)
        #only need one invalid index per graph 
        invalid_actions=torch.unique(
            torch.cat(
                (
                    add_idc[0][invalid_add_to_non_empty_ids],
                    add_idc[0][invalid_add_to_empty_ids],
                    conn_idc[0][invalid_conn_to_empty_ids],
                    conn_idc[0][invalid_conn_ids],
                    conn_idc[0][invalid_conn_self_ids],
                    conn_idc[0][invalid_conn_multi_ids],
                )
            )
        )
        """
        print ('invalid_add_to_non_empty_ids:',add_idc[0][invalid_add_to_non_empty_ids])
        print ('invalid_add_to_empty_ids:',add_idc[0][invalid_add_to_empty_ids])
        print ('invalid_conn_to_empty_ids:',conn_idc[0][invalid_conn_to_empty_ids])
        print ('invalid_conn_ids:',conn_idc[0][invalid_conn_ids])
        print ('invalid_conn_self_ids:',conn_idc[0][invalid_conn_self_ids])
        print ('invalid_conn_multi_ids:',conn_idc[0][invalid_conn_multi_ids])
        """
        
        invalid_add_to_impossibles=add_idc[0][invalid_add_to_impossible_ids]
        
        return invalid_actions, invalid_add_to_impossibles
    
    def pred_nconn_apd(self,g_nodes,g_edges,g_coords,g_masks,r_nodes,r_edges,focused_ids,r_masks,nconn_constrain_masks=None):
        softmax=torch.nn.Softmax(dim=1)
        #print ('focused_ids',focused_ids,focused_ids.shape)
        conn_pred=self.node_conn_model(g_nodes,g_edges,g_coords,g_masks,r_nodes,r_edges,focused_ids,r_masks)
        if nconn_constrain_masks is None:
            nconn_constrain_masks=torch.ones_like(conn_pred)
        conn_pred=softmax(conn_pred)*nconn_constrain_masks.to(conn_pred) 
        return conn_pred
    
    def split_nconn_action(self,action):
        conn_shape=(action.shape[0],*FGP.leaf_conn_dim)
        f_conn=action.view(conn_shape)
        conn_idc=torch.nonzero(f_conn,as_tuple=True)
        return conn_idc
    
    def nconn_step(self,g_nodes,g_edges,g_coords,g_masks,r_nodes,r_edges,focused_ids,r_masks,l_atomnums,nconn_constrain_masks=None,temp=1.0):
        
        conn_pred=self.pred_nconn_apd(g_nodes,g_edges,g_coords,g_masks,r_nodes,r_edges,focused_ids,r_masks,nconn_constrain_masks)
        conn_action=self.sample_actions(conn_pred,temp)
        likelihoods=self.compute_likelihoods(conn_pred,conn_action)
        
        conn_idc=self.split_nconn_action(conn_action)
        conn_from=l_atomnums[conn_idc[0]]+focused_ids[conn_idc[0]].view(-1)
        conn_idc=(*conn_idc,conn_from)
        
        return conn_idc,likelihoods
    
    def invalid_nconn_actions(self,conn_idc,l_atomnums):
        empty_graphs=torch.nonzero(l_atomnums[conn_idc[0]]==0)
        invalid_conn_ids=torch.nonzero(conn_idc[1]>=l_atomnums[conn_idc[0]])
        combined=torch.cat((invalid_conn_ids,empty_graphs),dim=0).squeeze(1)
        
        uniques,counts=combined.unique(return_counts=True)
        #invalid_conn_ids=uniques[counts==1].unsqueeze(dim=1)
        invalid_actions=conn_idc[0][invalid_conn_ids].view(-1)
        return invalid_actions
    
    def pred_nint_apd(self,complex_2D_nodes,complex_2D_edges,complex_2D_masks,
                 complex_3D_nodes,complex_3D_edges,complex_3D_coords,complex_3D_masks,
                 pgroups,pgroups_masks,pgroups_int_masks,focus_lgroups,focus_lgroups_masks,focus_ftypes,nint_constrain_masks=None):
        
        softmax=torch.nn.Softmax(dim=1)
        nint_pred=self.node_int_model(complex_2D_nodes,complex_2D_edges,complex_2D_masks,
                               complex_3D_nodes,complex_3D_edges,complex_3D_coords,complex_3D_masks,
                               pgroups,pgroups_masks,pgroups_int_masks,focus_lgroups,focus_lgroups_masks,focus_ftypes)
        
        if nint_constrain_masks is None:
            nint_constrain_masks=torch.ones_like(nint_pred)
        nint_pred=softmax(nint_pred)*nint_constrain_masks.to(nint_pred) 
        
        return nint_pred
    
    def split_nint_action(self,action):
        
        nint_shape=(action.shape[0],*FGP.leaf_int_add_dim)
        f_nint=action[:,:-1].view(nint_shape)
        nint_idc=torch.nonzero(f_nint,as_tuple=True)
        f_term=action[:,-1]
        term_idc=torch.nonzero(f_term,as_tuple=True)
        
        return nint_idc,term_idc
    
    def nint_step(self,complex_2D_nodes,complex_2D_edges,complex_2D_masks,
                 complex_3D_nodes,complex_3D_edges,complex_3D_coords,complex_3D_masks,
                 pgroups,pgroups_masks,pgroups_int_masks,focus_lgroups,focus_lgroups_masks,focus_ftypes,focus_lgroup_ids,nint_constrain_masks=None,temp=1.0):

        nint_pred=self.pred_nint_apd(complex_2D_nodes,complex_2D_edges,complex_2D_masks,
                               complex_3D_nodes,complex_3D_edges,complex_3D_coords,complex_3D_masks,
                               pgroups,pgroups_masks,pgroups_int_masks,focus_lgroups,focus_lgroups_masks,focus_ftypes,nint_constrain_masks)

        nint_action=self.sample_actions(nint_pred,temp)
        
        likelihoods=self.compute_likelihoods(nint_pred,nint_action)
        
        nint_idc,term_idc=self.split_nint_action(nint_action)

        nint_from=focus_lgroup_ids[nint_idc[0]]
        
        nint_idc=(*nint_idc,nint_from)
        
        return nint_idc,term_idc,likelihoods
    
    def invalid_nint_actions(self,nint_idc,n_pgroups):
        invalid_nint_ids=torch.nonzero(nint_idc[1]>=n_pgroups[nint_idc[0]])
        return invalid_nint_ids
    
     
    def __to_device(self,tensor):
        if self.local_rank is not None:
            return tensor.cuda(self.local_rank)
        else:
            return tensor.cuda()

    def __to_model_device(self,model):
        if self.local_rank is not None:
            model.cuda(self.local_rank)
            model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model=torch.nn.parallel.DistributedDataParallel(
                                                        model,
                                                        device_ids=[self.local_rank],
                                                        output_device=self.local_rank,
                                                        find_unused_parameters=True,
                                                        broadcast_buffers=False
                                                        )
        else:
            model.cuda()
        return model
    
    def __reset(self):
        self.conf_online_model=None
        self.conf_ema_model=None
        self.consistency_sampling_and_editing=None
        self.consistency_training=None
        self.node_add_model=None 
        self.ring_gen_model=None
        self.node_conn_model=None
        self.node_int_model=None
        self.optim=None
        self.lr_scheduler=None
        self.min_train_loss_epoch=1e20
        self.train_batch_loss=0
        self.train_epoch_loss=0
        self.min_valid_loss_epoch=1e20
        self.valid_batch_loss=0
        self.valid_epoch_loss=0
        self.logger_dict={}

        for key in ['coords','nadd','rgen','nconn','nint']:
            if key in self.jobs:
                self.logger_dict[key]=self.__create_logger(key)

    def __create_logger(self,mode):
        logger=open(f'./{self.modelname}/Training_{mode}.log','a')
        logger.write('='*40+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n') 
        logger.flush()
        return logger 

    def save_nadd_model(self,mode):
        if FGP.with_ctrlnet_for_nadd:
            savename=f"nadd_model_with_ctrlnet_{mode}.cpk"
        else:
            savename=f"nadd_model_{mode}.cpk"
        self.save_model_params(self.node_add_model,f'{self.modelname}/model/{savename}')
        return
    
    def save_rgen_model(self,mode):
        if FGP.with_ctrlnet_for_rgen:
            savename=f"rgen_model_with_ctrlnet_{mode}.cpk"
        else:
            savename=f"rgen_model_{mode}.cpk"
        self.save_model_params(self.ring_gen_model,f'{self.modelname}/model/{savename}')
        return
    
    def save_nconn_model(self,mode):
        if FGP.with_ctrlnet_for_nconn:
            savename=f"nconn_model_with_ctrlnet_{mode}.cpk"
        else:
            savename=f"nconn_model_{mode}.cpk"
        self.save_model_params(self.node_conn_model,f'{self.modelname}/model/{savename}')
        return
    
    def save_nint_model(self,mode):
        if FGP.with_ctrlnet_for_nint:
            savename=f"nint_model_with_ctrlnet_{mode}.cpk"
        else:
            savename=f"nint_model_{mode}.cpk"
        self.save_model_params(self.node_int_model,f'{self.modelname}/model/{savename}')
        return
    
    def save_conf_model(self,mode):
        self.save_model_params(self.conf_online_model,f'{self.modelname}/model/conf_online_model_{mode}.cpk')
        self.save_model_params(self.conf_ema_model,f'{self.modelname}/model/conf_ema_model_{mode}.cpk')
        return 
    
    def save_model_params(self,model,savepath):
        savedict={'lr':self.lr,'lossmin':self.min_valid_loss_epoch,'state_dict':model.state_dict()}
        torch.save(savedict,savepath)
        return
    
    def save_cpkt(self,mode):
        if 'coords' in self.jobs:
            self.save_conf_model(mode)
        if 'nadd' in self.jobs:
            self.save_nadd_model(mode)
        if  'rgen' in self.jobs:
            self.save_rgen_model(mode)
        if 'nconn' in self.jobs:
            self.save_nconn_model(mode)
        if 'nint' in self.jobs:
            self.save_nint_model(mode)
        return 

def load_state_dict_to_single(modelcpkt,parallel=True):
    para_sd={}
    for key in modelcpkt["state_dict"].keys():
        if parallel:
            para_sd['module.'+key.replace('module.','')]=modelcpkt["state_dict"][key]
        else:
            para_sd[key.replace('module.','')]=modelcpkt["state_dict"][key]

    return para_sd

def onek_encoding_unk(value, choices):
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices))
    index = choices.index(value)
    encoding[index] = 1
    return encoding
