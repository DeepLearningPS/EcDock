import torch

from .equiformer import * 
from .consistency import * 
from .en_transformer import * 
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
        
         
        self.optim=None
        self.lr_scheduler=None
        self.min_train_loss_epoch=1e20
        self.train_batch_loss=0
        self.train_epoch_loss=0
        self.min_valid_loss_epoch=1e20
        
        self.valid_batch_loss=0
        self.valid_epoch_loss=0
        if not os.path.exists(f'./{self.modelname}/model'):
            os.system(f'mkdir -p ./{self.modelname}/model')

        if self.mode=="train":
            self.__build_model()
        else:
            self.loadtype=kwargs.get("loadtype")
            self.load(self.modelname,self.loadtype)
            
        if 'coords' in self.jobs:
            self.logger_conf=open(f'./{self.modelname}/Training_coords.log','a')
            self.logger_conf.write('='*40+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n') 
            self.logger_conf.flush()
        if 'nadd' in self.jobs:
            self.logger_nadd=open(f'./{self.modelname}/Training_nadd.log','a')
            self.logger_nadd.write('='*40+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n') 
            self.logger_nadd.flush()
        if 'rgen' in self.jobs:
            self.logger_rgen=open(f'./{self.modelname}/Training_rgen.log','a')
            self.logger_rgen.write('='*40+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n') 
            self.logger_rgen.flush()
        if 'nconn' in self.jobs:
            self.logger_nconn=open(f'./{self.modelname}/Training_nconn.log','a')
            self.logger_nconn.write('='*40+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n') 
            self.logger_nconn.flush()
        if 'nint' in self.jobs:
            self.logger_nint=open(f'./{self.modelname}/Training_nint.log','a')
            self.logger_nint.write('='*40+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n') 
            self.logger_nint.flush()

        if epochs:
            self.epochs=epochs 

        return
    
    def __reset(self):
        self.conf_online_model=None
        self.conf_ema_model=None
        self.consistency_sampling_and_editing=None
        self.consistency_training=None
        self.node_add_model=None 
        self.ring_gen_model=None
        self.node_conn_model=None
        self.node_int_model=None
        self.logger_conf=None
        self.logger_nadd=None
        self.logger_rgen=None
        self.logger_nconn=None
        self.logger_nint=None
        self.optim=None
        self.lr_scheduler=None

    def __build_model(self):
        if self.local_rank is not None:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
        if 'coords' in self.jobs: 
            if FGP.conf_model_version=='v1':
                self.conf_online_model = Equiformer_Consistency(
                                                num_edge_tokens=len(FGP.bond_types)+1,
                                                num_tokens=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                                                edge_dim=4,
                                                dim = FGP.conf_dim,               # dimensions per type, ascending, length must match number of degrees (num_degrees)
                                                dim_head = FGP.conf_dim_head,          # dimension per attention head
                                                heads = FGP.conf_heads,             # number of attention heads
                                                num_linear_attn_heads = FGP.conf_num_linear_att_heads,     # number of global linear attention heads, can see all the neighbors
                                                num_degrees = FGP.conf_num_degrees,               # number of degrees
                                                depth = FGP.conf_depth,                     # depth of equivariant transformer
                                                attend_self = True,            # attending to self or not
                                                reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
                                                l2_dist_attention = False,      # set to False to try out MLP attention
                                                #reversible = True
                                                )
        
                self.conf_ema_model = Equiformer_Consistency(
                                                num_edge_tokens=len(FGP.bond_types)+1,
                                                num_tokens=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                                                edge_dim=4,
                                                dim = FGP.conf_dim,               # dimensions per type, ascending, length must match number of degrees (num_degrees)
                                                dim_head = FGP.conf_dim_head,          # dimension per attention head
                                                heads = FGP.conf_heads,             # number of attention heads
                                                num_linear_attn_heads = FGP.conf_num_linear_att_heads,     # number of global linear attention heads, can see all the neighbors
                                                num_degrees = FGP.conf_num_degrees,               # number of degrees
                                                depth = FGP.conf_depth,                     # depth of equivariant transformer
                                                attend_self = True,            # attending to self or not
                                                reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
                                                l2_dist_attention = False,      # set to False to try out MLP attention
                                                #reversible = True
                                                )
            else:
                self.conf_online_model=EquiformerV2(
                    feat_dim=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                    edge_dim=len(FGP.bond_types)+1,
                    num_layers=FGP.conf_depth,
                    num_heads=8,
                    ffn_hidden_channels=256,
                    edge_channels=64,
                    target="L2"
                )
                self.conf_ema_model=EquiformerV2(
                    feat_dim=len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types),
                    edge_dim=len(FGP.bond_types)+1,
                    num_layers=FGP.conf_depth,
                    num_heads=8,
                    ffn_hidden_channels=256,
                    edge_channels=64,
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
            self.node_add_model=self.__to_model_device(Node_adder_3d())
        if 'rgen' in self.jobs:
            self.ring_gen_model=self.__to_model_device(Ring_gener_3d())
        if 'nconn' in self.jobs:
            self.node_conn_model=self.__to_model_device(Node_conner_3d())
        if 'nint' in self.jobs:
            self.node_int_model=self.__to_model_device(Node_int_3d())

        return  
    
    def load(self,modelname,loadtype='per_epoch'):
        with tempfile.TemporaryDirectory() as dirpath:
            with zipfile.ZipFile(modelname + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)
            self.__build_model()
            if loadtype=='Minloss':
                self.load_cpkt(dirpath,'minloss')
            else:
                self.load_cpkt(dirpath,'perepoch')
            print ("Load Model successful")
        return 
    
    def load_cpkt(self,dirpath,mode):
        if 'coords' in self.jobs:
            self.conf_online_model=self.load_params_into_models(self.conf_online_model,cpkt_path=f"{dirpath}/model/conf_online_model_{mode}.cpk")
            print ("Load conf online model successfully!")
            
            self.conf_ema_model=self.load_params_into_models(self.conf_ema_model,cpkt_path=f"{dirpath}/model/conf_ema_model_{mode}.cpk")
            print ("Load conf ema model successfully!")

        if 'nadd' in self.jobs: 
            self.node_add_model=self.load_params_into_models(self.node_add_model,cpkt_path=f"{dirpath}/model/nadd_model_{mode}.cpk")
            print ("Load nadd model successfully!")
            
        if 'rgen' in self.jobs:
            self.ring_gen_model=self.load_params_into_models(self.ring_gen_model,cpkt_path=f"{dirpath}/model/rgen_model_{mode}.cpk")
            print ("Load rgen model successfully!")
            
        if 'nconn' in self.jobs:
            self.node_conn_model=self.load_params_into_models(self.node_conn_model,cpkt_path=f"{dirpath}/model/nconn_model_{mode}.cpk")
            print ("Load nconn model successfully!")
        if 'nint' in self.jobs:
            self.node_int_model=self.load_params_into_models(self.node_int_model,cpkt_path=f"{dirpath}/model/nint_model_{mode}.cpk")
            print ("Load nint model successfully!")    
        return 

    def load_params_into_models(self,model,cpkt_path):
        if os.path.exists(cpkt_path):
            modelcpkt=torch.load(cpkt_path,map_location=torch.device('cpu'))
            para_sd=load_state_dict_to_single(modelcpkt)
            model.load_state_dict(para_sd)
        else:
            print (f"Cannot find {cpkt_path} !")
            exit()
        return model 

    def save_cpkt(self,mode):
        if 'coords' in self.jobs:
            self.save_model_params(self.conf_online_model,f'{self.modelname}/model/conf_online_model_{mode}.cpk')
            self.save_model_params(self.conf_ema_model,f'{self.modelname}/model/conf_ema_model_{mode}.cpk')
        if 'nadd' in self.jobs:
            self.save_model_params(self.node_add_model,f'{self.modelname}/model/nadd_model_{mode}.cpk')
        if  'rgen' in self.jobs:
            self.save_model_params(self.ring_gen_model,f'{self.modelname}/model/rgen_model_{mode}.cpk')
        if 'nconn' in self.jobs:
            self.save_model_params(self.node_conn_model,f'{self.modelname}/model/nconn_model_{mode}.cpk')
        if 'nint' in self.jobs:
            self.save_model_params(self.node_int_model,f'{self.modelname}/model/nint_model_{mode}.cpk')
        return 
    
    def save_model_params(self,model,savepath):
        savedict={'lr':self.lr,'lossmin':self.min_valid_loss_epoch,'state_dict':model.state_dict()}
        torch.save(savedict,savepath)
        return
    
    def IC_Loss(self,pred,target,zbs,zas,zds,zbmasks,zamasks,zdmasks,gmasks):
        pred_bonddis,pred_angle,pred_dihedral=xyz2ics_v2(pred,zbs,zas,zds)
        target_bonddis,target_angle,target_dihedral=xyz2ics_v2(target,zbs,zas,zds)
        
        pred_dismat=torch.cdist(pred,pred,compute_mode='donot_use_mm_for_euclid_dist')
        target_dismat=torch.cdist(target,target,compute_mode='donot_use_mm_for_euclid_dist')
        
        gmasks_2D=(gmasks.unsqueeze(-1)*gmasks.unsqueeze(-1).permute(0,2,1)).bool()
        if torch.sum(gmasks_2D)>0:
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
    
    def Coord_Gen_Datas_to_gpu(self,Datas):
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
        
        return c_atoms,c_feats,c_adjs_mat,c_coords,c_fix_labels,c_flexible_labels,c_masks,\
            l_zbs,l_zas,l_zds,l_zbmasks,l_zamasks,l_zdmasks,\
            p_zbs,p_zas,p_zds,p_zbmasks,p_zamasks,p_zdmasks,\
            c_pocket_labels,c_ligand_labels
    
    def Train_Conf_Step(self,Datas,step_id,mode='train'):
        c_atoms,c_feats,c_adjs_mat,c_coords,c_fix_labels,c_flexible_labels,c_masks,\
            l_zbs,l_zas,l_zds,l_zbmasks,l_zamasks,l_zdmasks,\
                p_zbs,p_zas,p_zds,p_zbmasks,p_zamasks,p_zdmasks,\
                    c_pocket_labels,c_ligand_labels=self.Coord_Gen_Datas_to_gpu(Datas)
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
    
    def Leaf_Add_Datas_to_gpu(self,Datas):
        c_feats=self.__to_device(Datas["C_Feats"].float())
        c_adjs_mat=self.__to_device(Datas["C_Adjs_Mat"].float())
        c_coords=self.__to_device(Datas["C_Coords"].float())
        c_masks=self.__to_device(Datas["C_Masks"])
        l_apds=self.__to_device(Datas["L_Apds"].float())
        return c_feats,c_adjs_mat,c_coords,c_masks,l_apds
    
    def Train_Leaf_Add_Step(self,Datas,mode='train'):
        if mode!='train':
            self.node_add_model.eval()
        else:
            self.node_add_model.train()
            
        c_feats,c_adjs_mat,c_coords,c_masks,l_apds=self.Leaf_Add_Datas_to_gpu(Datas)
        self.optim.zero_grad()
        total_loss=0
        num=0
        for i in range(FGP.accsteps):
            num+=1
            if len(c_feats[i*FGP.batchsize:(i+1)*FGP.batchsize])>0:
                
                APD_pred=self.node_add_model(
                                                c_feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                c_adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                c_coords[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                c_masks[i*FGP.batchsize:(i+1)*FGP.batchsize]
                                            )
                
                nadd_loss=self.KL_Loss(APD_pred,l_apds[i*FGP.batchsize:(i+1)*FGP.batchsize])

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

    def Leaf_Rgen_Datas_to_gpu(self,Datas):
        c_feats=self.__to_device(Datas["C_Feats"].float())
        c_adjs_mat=self.__to_device(Datas["C_Adjs_Mat"].float())
        c_coords=self.__to_device(Datas["C_Coords"].float())
        c_masks=self.__to_device(Datas["C_Masks"])
        r_feats=self.__to_device(Datas["R_Feats"].float())
        r_adjs_mat=self.__to_device(Datas["R_Adjs_Mat"].float())
        r_masks=self.__to_device(Datas["R_Masks"])
        r_ftypes=self.__to_device(Datas["R_Ftypes"].float())
        r_apds=self.__to_device(Datas["R_Apds"].float())

        return c_feats,c_adjs_mat,c_coords,c_masks,r_feats,r_adjs_mat,r_masks,r_ftypes,r_apds

    def Train_Ring_Gen_Step(self,Datas,mode='train'):
        if mode!='train':
            self.ring_gen_model.eval()
        else:
            self.ring_gen_model.train()
            
        c_feats,c_adjs_mat,c_coords,c_masks,r_feats,r_adjs_mat,r_masks,r_ftypes,l_apds=self.Leaf_Rgen_Datas_to_gpu(Datas)
        
        self.optim.zero_grad()
        
        total_loss=0
        num=0
        for i in range(FGP.accsteps):
            num+=1
            
            if len(c_feats[i*FGP.batchsize:(i+1)*FGP.batchsize])>0:
                APD_pred=self.ring_gen_model(
                                                c_feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                c_adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                c_coords[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                c_masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                r_feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                r_adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                r_ftypes[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                r_masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                            )
                
                rgen_loss=self.KL_Loss(APD_pred,l_apds[i*FGP.batchsize:(i+1)*FGP.batchsize])
                
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
    
    def Leaf_Conn_Datas_to_gpu(self,Datas):
        c_feats=self.__to_device(Datas["C_Feats"].float())
        c_adjs_mat=self.__to_device(Datas["C_Adjs_Mat"].float())
        c_coords=self.__to_device(Datas["C_Coords"].float())
        c_masks=self.__to_device(Datas["C_Masks"])
        r_feats=self.__to_device(Datas["R_Feats"].float())
        r_adjs_mat=self.__to_device(Datas["R_Adjs_Mat"].float())
        r_masks=self.__to_device(Datas["R_Masks"])
        focus_atom=self.__to_device(Datas["Focus_Atom"].float())
        l_apds=self.__to_device(Datas["L_Apds"].float())
        return c_feats,c_adjs_mat,c_coords,c_masks,r_feats,r_adjs_mat,r_masks,focus_atom,l_apds
    
    def Train_Node_Conn_Step(self,Datas,mode='train'):
        if mode!='train':
            self.node_conn_model.eval()
        else:
            self.node_conn_model.train()
            
        c_feats,c_adjs_mat,c_coords,c_masks,r_feats,r_adjs_mat,r_masks,focus_atom,l_apds=self.Leaf_Conn_Datas_to_gpu(Datas)
        self.optim.zero_grad()
        total_loss=0
        num=0
        for i in range(FGP.accsteps):
            num+=1
            if len(c_feats[i*FGP.batchsize:(i+1)*FGP.batchsize])>0:
                APD_pred=self.node_conn_model(
                                                c_feats[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                c_adjs_mat[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                c_coords[i*FGP.batchsize:(i+1)*FGP.batchsize],
                                                c_masks[i*FGP.batchsize:(i+1)*FGP.batchsize],
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
                
                nint_loss=self.KL_Loss(APD_pred,l_apds[i*FGP.batchsize:(i+1)*FGP.batchsize])

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

    def Fit(self,Complex_MGFiles,Drugs_MGFiles,Epochs=100,split_rate=0.95,mode='coords',nfiles_per_mini_epochs=1000):
        if mode=='coords':
            self.optim=Adam(self.conf_online_model.parameters(), lr = FGP.initlr, betas=(0.5,0.999))
        elif mode=='nadd':
            self.optim=Adam(self.node_add_model.parameters(), lr = FGP.initlr, betas=(0.5,0.999))
        elif mode=='rgen':
            self.optim=Adam(self.ring_gen_model.parameters(), lr = FGP.initlr, betas=(0.5,0.999))
        elif mode=='nconn':
            self.optim=Adam(self.node_conn_model.parameters(), lr = FGP.initlr, betas=(0.5,0.999))
        elif mode=='nint':
            self.optim=Adam(self.node_int_model.parameters(), lr = FGP.initlr, betas=(0.5,0.999))
        self.lr_scheduler= ReduceLROnPlateau(
                self.optim, mode='min',
                factor=0.9, patience=FGP.lr_patience,
                verbose=True, threshold=0.0001, threshold_mode='rel',
                cooldown=FGP.lr_cooldown,
                min_lr=1e-06, eps=1e-06)
        n_mini_epochs=math.ceil(len(Complex_MGFiles)/nfiles_per_mini_epochs)
        
        num=0
        for epoch in range(Epochs):
            for mini_epoch in range(n_mini_epochs):
                
                Complex_MGs=[]
                for Fname in Complex_MGFiles[mini_epoch*nfiles_per_mini_epochs:(mini_epoch+1)*nfiles_per_mini_epochs]:
                    with open(Fname,'rb') as f:
                        comp=pickle.load(f)
                        Complex_MGs.append(comp)
                
                Drug_MGs=[]
                if len(Drugs_MGFiles)>0:
                    DrugFiles=random.samples(Drugs_MGFiles,nfiles_per_mini_epochs)
                
                    for Fname in DrugFiles:
                        with open(Fname,'rb') as f:
                            drug=pickle.load(f)
                            Drug_MGs.append(drug)

                MGs=Complex_MGs+Drug_MGs
                cutnum=math.ceil(len(MGs)*split_rate)
                
                if self.local_rank is not None:
                    Train_Dataset=MG_Dataset(MGs[:cutnum],name='trainset',mode=mode)
                    train_sampler=torch.utils.data.distributed.DistributedSampler(Train_Dataset)
                    trainloader=DataLoader(Train_Dataset,batch_size=self.batchsize*FGP.accsteps,shuffle=False,num_workers=FGP.n_workers,sampler=train_sampler)
                    train_sampler.set_epoch(epoch)
                    
                    Valid_Dataset=MG_Dataset(MGs[cutnum:],name='validset',mode=mode)
                    valid_sampler=torch.utils.data.distributed.DistributedSampler(Valid_Dataset)
                    validloader=DataLoader(Valid_Dataset,batch_size=self.batchsize*FGP.accsteps,shuffle=False,num_workers=FGP.n_workers,sampler=valid_sampler)
                    valid_sampler.set_epoch(epoch)
                    
                else:
                    Train_Dataset=MG_Dataset(MGs[:cutnum],name='trainset',mode=mode)
                    trainloader=DataLoader(Train_Dataset,batch_size=self.batchsize*FGP.accsteps,shuffle=False,num_workers=FGP.n_workers)
                    Valid_Dataset=MG_Dataset(MGs[cutnum:],name='validset',mode=mode)
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
                            step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_loss_xyz,step_lstr=self.Train_Conf_Step(Datas,step_id=step,mode='train')
                            lstr=f'Training -- Epochs: {epoch},{mini_epoch} bid: {bid} step: {step} lr: {self.lr:.3E} '+step_lstr
                            print (lstr)
                            self.logger_conf.write(lstr+'\n')
                            self.logger_conf.flush()
                            train_batch_loss+=step_loss
                            
                        self.train_batch_loss=train_batch_loss
                        self.lr_scheduler.step(metrics=self.train_batch_loss)
                        self.train_epoch_loss+=train_batch_loss
                        torch.cuda.empty_cache() 
                        
                    elif mode=='nadd':
                        step_loss,step_lstr=self.Train_Leaf_Add_Step(Datas,mode='train')
                        lstr=f'Training -- Epochs: {epoch},{mini_epoch} bid: {bid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_nadd.write(lstr+'\n')
                        self.logger_nadd.flush()
                        self.train_batch_loss=step_loss
                        self.lr_scheduler.step(metrics=self.train_batch_loss)
                        torch.cuda.empty_cache()
                        
                    elif mode=='rgen':
                        step_loss,step_lstr=self.Train_Ring_Gen_Step(Datas,mode='train')
                        lstr=f'Training -- Epochs: {epoch},{mini_epoch} bid: {bid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_rgen.write(lstr+'\n')
                        self.logger_rgen.flush()
                        self.train_batch_loss=step_loss
                        self.lr_scheduler.step(metrics=self.train_batch_loss)
                        torch.cuda.empty_cache()
                        
                    elif mode=='nconn':
                        step_loss,step_lstr=self.Train_Node_Conn_Step(Datas,mode='train')
                        lstr=f'Training -- Epochs: {epoch},{mini_epoch} bid: {bid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_nconn.write(lstr+'\n')
                        self.logger_nconn.flush()
                        self.train_batch_loss=step_loss
                        self.lr_scheduler.step(metrics=self.train_batch_loss)
                        torch.cuda.empty_cache()
                        
                    elif mode=='nint':
                        step_loss,step_lstr=self.Train_Node_Int_Step(Datas,mode='train')
                        lstr=f'Training -- Epochs: {epoch},{mini_epoch} bid: {bid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_nint.write(lstr+'\n')
                        self.logger_nint.flush()
                        self.train_batch_loss=step_loss
                        self.lr_scheduler.step(metrics=self.train_batch_loss)
                        torch.cuda.empty_cache()
                        
                self.valid_epoch_loss=0

                for vid,vDatas in validbar:
                    valid_batch_loss=0
                    if mode=='coords':
                        for step in range(FGP.final_timesteps):
                            with torch.no_grad():
                                step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_loss_xyz,step_lstr=self.Train_Conf_Step(vDatas,step_id=step,mode='eval')
                            lstr=f'Valid    -- Epochs: {epoch},{mini_epoch} bid: {vid} step: {step} lr: {self.lr:.3E} '+step_lstr
                            print (lstr+'\n')
                            self.logger_conf.write(lstr+'\n')
                            self.logger_conf.flush()
                            valid_batch_loss+=step_loss 
                        self.valid_epoch_loss+=valid_batch_loss
                        self.logger_conf.write(f'{Fname} Valid    -- Epochs: validloss: {self.valid_epoch_loss/nvalid_batchs:.3E}') 
                        self.logger_conf.flush()
                        torch.cuda.empty_cache() 
                    if mode=='nadd':
                        with torch.no_grad():
                            step_loss,step_lstr=self.Train_Leaf_Add_Step(vDatas,mode='eval')
                        lstr=f'Valid    -- Epochs: {epoch},{mini_epoch} bid: {vid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_nadd.write(lstr+'\n')
                        self.logger_nadd.flush()
                        valid_batch_loss+=step_loss
                        self.valid_epoch_loss+=valid_batch_loss
                        torch.cuda.empty_cache()
                    elif mode=='rgen':
                        with torch.no_grad():
                            step_loss,step_lstr=self.Train_Ring_Gen_Step(vDatas,mode='eval')
                        lstr=f'Valid    -- Epochs: {epoch},{mini_epoch} bid: {vid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_rgen.write(lstr+'\n')
                        self.logger_rgen.flush()
                        valid_batch_loss+=step_loss
                        self.valid_epoch_loss+=valid_batch_loss
                        torch.cuda.empty_cache()
                    elif mode=='nconn':
                        with torch.no_grad():
                            step_loss,step_lstr=self.Train_Node_Conn_Step(vDatas,mode='eval')
                        lstr=f'Valid    -- Epochs: {epoch},{mini_epoch} bid: {vid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_nconn.write(lstr+'\n')
                        self.logger_nconn.flush()
                        valid_batch_loss+=step_loss
                        self.valid_epoch_loss+=valid_batch_loss
                        torch.cuda.empty_cache()
                    elif mode=='nint':
                        with torch.no_grad():
                            step_loss,step_lstr=self.Train_Node_Int_Step(vDatas,mode='eval')
                        lstr=f'Valid    -- Epochs: {epoch},{mini_epoch} bid: {vid} lr: {self.lr:.3E} '+step_lstr
                        print (lstr)
                        self.logger_nint.write(lstr+'\n')
                        self.logger_nint.flush()
                        valid_batch_loss+=step_loss
                        self.valid_epoch_loss+=valid_batch_loss
                        torch.cuda.empty_cache()
                print (self.valid_epoch_loss,self.min_valid_loss_epoch)
                if self.valid_epoch_loss<self.min_valid_loss_epoch:
                    self.min_valid_loss_epoch=self.valid_epoch_loss
                    print (f'Save New check point of online & ema model at Epoch:{epoch},{mini_epoch}')
                    if self.local_rank==0 or self.local_rank is None:
                        self.save_cpkt(mode='minloss')

                if self.local_rank==0 or self.local_rank is None:
                    self.save_cpkt(mode='perepoch')
                num+=1
                if self.local_rank is not None:
                    dist.barrier()
        return 
    
    def sample_coords_batch(self,Datas):
        c_atoms,c_feats,c_adjs_mat,c_coords,c_fix_labels,c_flexible_labels,c_masks,c_zbs,c_zas,c_zds,c_zbmasks,c_zamasks,c_zdmasks,c_pocket_labels,c_ligand_labels=self.Coord_Gen_Datas_to_gpu(Datas)
        
        with torch.no_grad():
            sigmas = karras_schedule(
                FGP.final_timesteps, FGP.sigma_min, FGP.sigma_max, FGP.rho, c_coords.device
            )
            sigmas= reversed(sigmas)[:-1]

            samples,sample_diff, sample_diff_before= self.consistency_sampling_and_editing(
                                    self.conf_online_model,
                                    atoms=c_atoms,
                                    feats=c_feats,
                                    adjs=c_adjs_mat,
                                    ligand_labels=c_ligand_labels,
                                    y=torch.randn_like(c_coords).to(c_coords), # used to infer the shapes
                                    gmasks=c_masks,
                                    sigmas=sigmas, # sampling starts at the maximum std (T)
                                    clip_denoised=False, # whether to clamp values to [-1, 1] range
                                    verbose=True,
                                )
        
        return samples ,sample_diff, sample_diff_before
    
    def inpaint_coords_batch(self,c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,c_fix_labels,c_flexible_labels,c_pocket_labels,c_ligand_labels):
        # coords generator fails in 3 aspect, 
        # 1. the pocket_labels should be provided, when ligand labels provided, the mass center is incorrect for docking model.
        # 2. the c_flexible_masks should be zeros in ghost padding positions
        # 3. the pl_int_adjs should be atom-based interaction adjs instead of group-divided atom adjs. 
        #c_atoms,c_feats,c_adjs_mat,c_coords,c_fix_labels,c_flexible_labels,c_masks,c_zbs,c_zas,c_zds,c_zbmasks,c_zamasks,c_zdmasks,c_pocket_labels,c_ligand_labels=self.Coord_Gen_Datas_to_gpu(Datas)
        
        inpaint_coords=c_coords*c_fix_labels.unsqueeze(-1).float()
        print ('coords',inpaint_coords.type())
        #print (c_flexible_labels)
        with torch.no_grad():
            sigmas = karras_schedule(
                FGP.final_timesteps, FGP.sigma_min, FGP.sigma_max, FGP.rho, c_coords.device
            )
            sigmas= self.__to_device(reversed(sigmas)[:-1]).float()
            print ('flexible_labels',c_flexible_labels)
            print (sigmas.type())
            samples,samples_diff,sample_diff_before = self.consistency_sampling_and_editing(
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
                                )

        return samples,samples_diff,sample_diff_before 
    
    def Check_Conf_Model(self,MG,conf_num_per_states=10,batchsize=1,savepath='./sample',mode='complex'):

        Dataset=MG_Dataset([MG],name='sample',mode='coords')
        print (Dataset.__len__())
        Dataset.repulicate(rep_num=conf_num_per_states)
        print (MG.l_groups)
        print (Dataset.__len__())
        loader=DataLoader(Dataset,batch_size=batchsize,shuffle=False,num_workers=FGP.n_workers)
        bar=enumerate(loader)
        total_samples=[]
        total_samples_diff=[]
        total_samples_diff_before=[]
        for bid,Datas in bar:
            c_atoms,c_feats,c_adjs_mat,c_coords,c_fix_labels,c_flexible_labels,c_masks,\
                l_zbs,l_zas,l_zds,l_zbmasks,l_zamasks,l_zdmasks,\
                    p_zbs,p_zas,p_zds,p_zbmasks,p_zamasks,p_zdmasks,\
                    c_pocket_labels,c_ligand_labels=self.Coord_Gen_Datas_to_gpu(Datas) 
            samples,samples_diff,samples_diff_before=self.inpaint_coords_batch(\
                                    c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,\
                                    c_fix_labels,c_flexible_labels,c_pocket_labels,c_ligand_labels)
            
            samples_diff=torch.concat(samples_diff,axis=1)
            samples_diff_before=torch.concat(samples_diff_before,axis=1)
            #print (samples_diff.shape,samples_diff_before.shape)    
            total_samples.append(samples)
            total_samples_diff.append(samples_diff)
            total_samples_diff_before.append(samples_diff_before) 
        
        total_samples=torch.concat(total_samples,axis=0).clone().detach().cpu().numpy()
        total_samples_diff=torch.concat(total_samples_diff,axis=0).clone().detach().cpu().numpy()
        total_samples_diff_before=torch.concat(total_samples_diff_before,axis=0).clone().detach().cpu().numpy()
        print (total_samples.shape)
        print (total_samples_diff.shape)
        print (total_samples_diff_before.shape)
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
            print (l_natoms)
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
       
    def init_l_graphs(self,ids=None,mode='all'):  
        if mode!='all':
            assert ids is not None, 'ids should be provided when mode is not all'
        
        if mode=='all':             
            self.l_feats=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types))).float()
            self.l_adjs_mat=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,FGP.max_latoms,len(FGP.bond_types)+1)).float()
            self.l_coords=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms,3).float())
            self.l_atomnums=self.__to_device(torch.zeros(FGP.batchsize).long())
            self.l_atoms=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_gmasks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_dmasks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_rounds=self.__to_device(torch.zeros(FGP.batchsize).long())
            self.l_groups=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroups,FGP.max_lgroup_size).long())
            self.l_groups_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroups,FGP.max_lgroup_size).float())
            self.pl_adjs=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_patoms,FGP.max_latoms,len(FGP.bond_types)+1).float())
            self.l_likelihoods=self.__to_device(torch.zeros(FGP.batchsize).float())
            self.l_fix_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
            self.l_flexible_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_latoms).long())
        else:
            self.l_feats[ids]=0
            self.l_adjs_mat[ids]=0
            self.l_atomnums[ids]=0
            self.l_atoms[ids]=0
            self.l_masks[ids]=0
            self.l_gmasks[ids]=0
            self.l_dmasks[ids]=0
            self.l_rounds[ids]=0
            self.l_groups[ids]=0
            self.l_groups_masks[ids]=0
            self.pl_adjs[ids]=0
            self.l_likelihoods[ids]=0
            self.l_fix_masks[ids]=0
            self.l_flexible_masks[ids]=0
        return 
      
    def init_r_graphs(self,ids=None,mode='all'):
        if mode!='all':
            assert ids is not None, 'ids should be provided when mode is not all'
            
        if mode=='all':
            self.r_feats=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size,len(FGP.atom_types_for_feats)+len(FGP.formal_charge_types)).float())
            self.r_adjs_mat=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size,FGP.max_lgroup_size,len(FGP.bond_types)+1).float())
            self.r_atomnums=self.__to_device(torch.zeros(FGP.batchsize).long())
            self.r_atoms=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size).long())
            self.r_focused_ids=self.__to_device(torch.zeros(FGP.batchsize,1).long())
            self.r_masks=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size).long())
            self.r_rounds=self.__to_device(torch.zeros(FGP.batchsize)).long()
            self.r_ftypes=self.__to_device(torch.zeros(FGP.batchsize,FGP.n_group_feats)).float()
            self.r_likelihoods=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_lgroup_size*4).float())
            self.r_maxatoms=self.__to_device(torch.zeros(FGP.batchsize).long())
            
        else:
            
            self.r_feats[ids]=0
            self.r_adjs_mat[ids]=0
            self.r_atomnums[ids]=0
            self.r_atoms[ids]=0
            self.r_focused_ids[ids]=0
            self.r_masks[ids]=0
            self.r_rounds[ids]=0
            self.r_likelihoods[ids]=0
            if mode!='failed':
                self.r_ftypes[ids]=0
                self.r_maxatoms[ids]=0
                
        return 
    
    def init_c_graphs(self,Datas):
        self.c_feats,self.c_adjs_mat,self.c_coords,self.c_masks=\
            (self.__to_device(Datas[key]).float() for key in ["C_Feats","C_Adjs_Mat","C_Coords","C_Masks"])
        self.c_gmasks,self.c_dmasks=self.__to_device(Datas["C_GMasks"]),self.__to_device(Datas["C_DMasks"])
        self.p_groups=self.__to_device(Datas["P_Groups"]).long()
        self.p_groups_masks=self.__to_device(Datas["P_Groups_Masks"]).long()
        self.n_pgroups=self.__to_device(Datas["N_PGroups"]).long().view(-1)
        self.c_fix_masks=self.__to_device(Datas["C_Fix_Masks"]).long()
        self.c_flexible_masks=self.__to_device(Datas["C_Flexible_Masks"]).long()
        self.c_atoms=self.__to_device(Datas["C_Atoms"]).long()
        self.c_pocket_labels=self.__to_device(Datas["C_Pocket_Labels"]).long()
        self.c_ligand_labels=self.__to_device(torch.zeros(FGP.batchsize,FGP.max_patoms+FGP.max_latoms)).long()
        self.p_groups_int_masks=self.__to_device(Datas["P_INT_Groups_Masks"]).long()
        self.p_groups_atom_int_masks=self.__to_device(Datas["P_INT_Groups_Atom_Masks"]).long()
        self.p_atom_int_masks=self.__to_device(Datas["P_INT_Atom_Masks"]).long()
        return 
        
    def init_nconn(self,ids=None,mode='all'):
        if mode !='all':
            assert ids is not None, 'ids should be provided when mode is not all'
            
        if mode== 'all':
            self.nconn_likelihoods=self.__to_device(torch.zeros(FGP.batchsize).float())
            
        else:
            self.nconn_likelihoods[ids]=0
            
        return 
        
    def init_nint(self,ids=None,mode='all'):
        if mode!='all':
            assert ids is not None, 'ids should be provided when mode is not all'
        if mode=='all':
            self.nint_likelihoods=self.__to_device(torch.zeros(FGP.batchsize).float())
        else:
            self.nconn_likelihoods[ids]=0
        return
    
    def init_nadd(self,ids=None,mode='all'):
        if mode !='all':
            assert ids is not None, 'ids should be provided when mode is not all'
        if mode=='all':
            self.nadd_likelihoods=self.__to_device(torch.zeros(FGP.batchsize).float())
        else:
            self.nadd_likelihoods[ids]=0
        return
        
    def select_l_graphs(self,ids):
        l_feats=self.l_feats[ids]
        l_adjs_mat=self.l_adjs_mat[ids]
        l_coords=self.l_coords[ids]
        l_atomnums=self.l_atomnums[ids]
        l_atoms=self.l_atoms[ids]
        l_masks=self.l_masks[ids]
        l_rounds=self.l_rounds[ids]
        return l_feats,l_adjs_mat,l_coords,l_atomnums,l_atoms,l_masks,l_rounds
        
    def select_r_graphs(self,ids):
        
        r_feats=self.r_feats[ids]
        r_adjs_mat=self.r_adjs_mat[ids]
        r_atomnums=self.r_atomnums[ids]
        r_atoms=self.r_atoms[ids]
        r_masks=self.r_masks[ids]
        r_rounds=self.r_rounds[ids]
        r_ftypes=self.r_ftypes[ids]
        r_maxatoms=self.r_maxatoms[ids]
        r_focused_ids=self.r_focused_ids[ids]
        
        return r_feats,r_adjs_mat,r_atomnums,r_atoms,r_ftypes,r_masks,r_rounds,r_maxatoms,r_focused_ids
    
    def update_c_graphs(self):
        
        self.c_feats[:,FGP.max_patoms:]=self.l_feats
        self.c_atoms[:,FGP.max_patoms:]=self.l_atoms
        self.c_adjs_mat[:,FGP.max_patoms:,FGP.max_patoms:]=self.l_adjs_mat
        self.c_adjs_mat[:,:FGP.max_patoms,FGP.max_patoms:]=self.pl_adjs
        self.c_adjs_mat[:,FGP.max_patoms:,:FGP.max_patoms]=self.pl_adjs.transpose(1,2)
        lp_conn_adjs=(self.__to_device(torch.zeros_like(self.pl_adjs[:,:,:,0]))*self.p_atom_int_masks.unsqueeze(-1)).transpose(1,2)
        lp_conn_adjs=lp_conn_adjs*self.l_masks.unsqueeze(-1)
        
        self.c_adjs_mat[:,:FGP.max_patoms,FGP.max_patoms:,-1]=lp_conn_adjs.transpose(1,2)
        self.c_adjs_mat[:,FGP.max_patoms:,:FGP.max_patoms,-1]=lp_conn_adjs
        
        self.c_coords[:,FGP.max_patoms:]=self.l_coords
        self.c_masks[:,FGP.max_patoms:]=self.l_masks
        self.c_gmasks[:,FGP.max_patoms:]=self.l_gmasks
        self.c_dmasks[:,FGP.max_patoms:]=self.l_dmasks
        
        self.c_fix_masks[:,FGP.max_patoms:]=self.l_fix_masks
        self.c_flexible_masks[:,FGP.max_patoms:]=self.l_flexible_masks
        self.c_ligand_labels[:,FGP.max_patoms:]=self.l_masks
        
        return 
    
    def select_c_graphs(self,ids,mask_mode='normal',output_mode='simple'):
        if mask_mode=='graph':
            masks=self.c_gmasks
        elif mask_mode=='coords':
            masks=self.c_dmasks
        else:
            masks=self.c_masks
        
        self.update_c_graphs()
        c_feats=self.c_feats*masks.unsqueeze(-1).float()
        c_atoms=self.c_atoms*masks
        masks2D=(masks.unsqueeze(-1)*masks.unsqueeze(-2)).unsqueeze(-1)
        c_adjs_mat=self.c_adjs_mat*masks2D
        c_coords=self.c_coords*masks.unsqueeze(-1).float()
        c_pocket_labels=self.c_pocket_labels[ids]
        c_ligand_labels=self.c_ligand_labels[ids]
        c_feats=c_feats[ids]
        c_atoms=self.c_atoms[ids]
        c_adjs_mat=c_adjs_mat[ids]
        c_coords=c_coords[ids]
        c_masks=masks[ids]
        p_groups=self.p_groups[ids]
        p_groups_masks=self.p_groups_masks[ids]
        c_flexible_masks=self.c_flexible_masks[ids]
        c_fix_masks=self.c_fix_masks[ids]
        n_pgroups=self.n_pgroups[ids]
        
        p_groups_int_masks=self.p_groups_int_masks[ids]
        
        p_groups_atom_int_masks=self.p_groups_atom_int_masks[ids]
        
        p_atom_int_masks=self.p_atom_int_masks[ids]
        
        if output_mode=='simple':
            return c_feats,c_adjs_mat,c_coords,c_masks
        elif output_mode=='middle':
            return c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,p_groups,p_groups_masks,n_pgroups,p_groups_int_masks,p_atom_int_masks
        else:
            return c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,c_fix_masks,c_flexible_masks,p_groups,p_groups_masks,n_pgroups,p_groups_int_masks,p_groups_atom_int_masks,p_atom_int_masks,c_pocket_labels,c_ligand_labels
    
    def sample_prepare(self):
        self.l_stop_mask=self.__to_device(torch.ones(FGP.batchsize).long())
        self.r_stop_mask=self.__to_device(torch.ones(FGP.batchsize).long())
        self.nint_stop_mask=self.__to_device(torch.ones(FGP.batchsize).long())
        self.gid_to_gfeats=[]
        for i in range(len(FGP.group_index_type_dict.keys())):
            self.gid_to_gfeats.append(group_descriptor_to_puzzled_group_feats(FGP.group_index_type_dict[i]))
        self.gid_to_gfeats=np.array(self.gid_to_gfeats)
        self.gid_to_gfeats=self.__to_device(torch.Tensor(self.gid_to_gfeats)).float()
        self.single_atom_node_feats=[]
        for i in range(len(FGP.atom_types)):
            self.single_atom_node_feats.append(onek_encoding_unk(FGP.atom_types[i],FGP.atom_types_for_feats)+onek_encoding_unk(0,FGP.formal_charge_types))
        self.single_atom_node_feats=self.__to_device(torch.Tensor(self.single_atom_node_feats).float())
        return
    
    def update_r_maxatoms(self):
        r_ids=torch.nonzero(self.r_ftypes[:,0]).view(-1)
        print (r_ids)
        self.r_maxatoms[r_ids]=torch.sum(self.r_ftypes[r_ids,1:10],dim=1).long()
        single_ids=torch.nonzero(torch.sum(self.r_ftypes[:,1:10],dim=1)==1)
        print (single_ids)
        self.r_maxatoms[single_ids]=1
        pharm_ids=torch.nonzero(torch.sum(self.r_ftypes[:,1:10],dim=1)<0)
        print (pharm_ids)
        self.r_maxatoms[pharm_ids]=self.r_ftypes[pharm_ids,10].long()
        print (self.r_maxatoms)
        return 
    
    def update_single_atom_r_graphs(self,nadd_mol_ids,nadd_type_ids,single_node_ids):
        # nadd_mol_ids should correspond to all graphs
        
        self.r_feats[nadd_mol_ids[single_node_ids],0]=self.single_atom_node_feats[nadd_type_ids[single_node_ids]]
        self.r_atomnums[nadd_mol_ids[single_node_ids]]+=1
        self.r_atoms[nadd_mol_ids[single_node_ids],0]=self.__to_device(torch.Tensor(FGP.atom_types).long())[nadd_type_ids[single_node_ids]]
        return 
    
    def select_valid_r_add_actions(self,r_add_ids,valid_ids,likelihoods):
        # the valid_ids should corresponds to r_add_ids
        
        (batch,bond_to,atom_type,charge,bond_type,bond_from)=r_add_ids
        
        batch=batch[valid_ids]
        bond_to=bond_to[valid_ids]
        bond_from=bond_from[valid_ids]
        atom_type=atom_type[valid_ids]
        charge=charge[valid_ids]
        bond_type=bond_type[valid_ids] 
        likelihoods=likelihoods[valid_ids]
        
        return batch,bond_to,atom_type,charge,bond_type,bond_from,likelihoods
    
    def update_r_graphs_with_add_actions(self,batch,bond_to,atom_type,charge,bond_type,bond_from,likelihoods):
        # the batch should correspond to the ids to self.r_graphs
        
        self.r_feats[batch,bond_from,atom_type+2]=1
        self.r_feats[batch,bond_from,charge+len(FGP.atom_types_for_feats)]=1
        
        non_empty_r_ids=torch.nonzero(self.r_atomnums[batch]>0)
        batch_non_empty=batch[non_empty_r_ids]
        bond_to_non_empty=bond_to[non_empty_r_ids]
        bond_from_non_empty=bond_from[non_empty_r_ids]
        bond_type_non_empty=bond_type[non_empty_r_ids]
        
        self.r_atomnums[batch]+=1
        self.r_atoms[batch,bond_from]=\
            self.__to_device(torch.Tensor(FGP.atom_types).long())[atom_type]
        
        self.r_adjs_mat[batch_non_empty,
                        bond_from_non_empty,
                        bond_to_non_empty,
                        bond_type_non_empty+1]=1
        
        self.r_adjs_mat[batch_non_empty,
                        bond_to_non_empty,
                        bond_from_non_empty,
                        bond_type_non_empty+1]=1
        
        self.r_likelihoods[batch,self.r_rounds[batch]]=likelihoods
        self.r_rounds[batch]+=1
        return
    
    def select_valid_r_conn_actions(self,r_conn_ids,valid_ids,likelihoods):
        batch,bond_to,bond_type,bond_from=r_conn_ids
        batch=batch[valid_ids]
        bond_to=bond_to[valid_ids]
        bond_from=bond_from[valid_ids]
        bond_type=bond_type[valid_ids]
        likelihoods=likelihoods[valid_ids]
        return batch,bond_to,bond_type,bond_from,likelihoods

    def update_r_graphs_with_conn_actions(self,batch,bond_to,bond_type,bond_from,likelihoods):
        self.r_adjs_mat[batch,bond_from,bond_to,bond_type+1]=1
        self.r_adjs_mat[batch,bond_to,bond_from,bond_type+1]=1
        self.r_likelihoods[batch,self.r_rounds[batch]]=likelihoods
        self.r_rounds[batch]+=1
        return
    
    def select_valid_nconn_actions(self,nconn_ids,valid_ids,likelihoods):
        batch,bond_to,bond_type,bond_from=nconn_ids
        print (batch,bond_to,bond_type,bond_from,likelihoods)
        batch=batch[valid_ids]
        bond_to=bond_to[valid_ids]
        bond_from=bond_from[valid_ids]
        bond_type=bond_type[valid_ids]
        likelihoods=likelihoods[valid_ids]
        print (batch,bond_to,bond_type,bond_from,likelihoods)
        return batch,bond_to,bond_type,bond_from,likelihoods
    
    def update_l_graphs_with_nconn_actions(self,batch,bond_to,bond_type,bond_from,likelihoods):
        self.l_adjs_mat[batch,bond_from,bond_to,bond_type+1]=1
        self.l_adjs_mat[batch,bond_to,bond_from,bond_type+1]=1
        self.nconn_likelihoods[batch]=likelihoods
        return 
    
    def update_l_graphs_with_rgraphs(self,ids):
        for i in ids:
            l_natoms=self.l_atomnums[i]
            r_natoms=self.r_atomnums[i]
            self.l_feats[i,l_natoms:l_natoms+r_natoms]=self.r_feats[i,:r_natoms]
            self.l_adjs_mat[i,l_natoms:l_natoms+r_natoms,l_natoms:l_natoms+r_natoms]=self.r_adjs_mat[i,:r_natoms,:r_natoms]
            self.l_atoms[i,l_natoms:l_natoms+r_natoms]=self.r_atoms[i,:r_natoms]
            self.l_groups[i,self.l_rounds[i],:r_natoms]=l_natoms+self.__to_device(torch.arange(r_natoms).long())
            self.l_groups_masks[i,self.l_rounds[i],:r_natoms]=1
            self.l_masks[i,:l_natoms+r_natoms]=1
            self.l_gmasks[i,l_natoms:l_natoms+r_natoms]=1
            self.l_atomnums[i]+=r_natoms
            self.l_rounds[i]+=1            
            
            self.l_fix_masks[i]=0
            self.l_flexible_masks[i]=0
            if FGP.l_mode=='fix-previous':
                self.l_fix_masks[i,:l_natoms]=1
                self.l_fix_masks[i,l_natoms:self.l_atomnums[i]]=1
            else:    
                self.l_fix_masks[i,:self.l_atomnums[i]]=0
                self.l_flexible_masks[i,:self.l_atomnums[i]]=1
        return 
    
    def select_valid_nint_actions(self,nint_ids,valid_ids,likelihoods):
        batch,int_to,int_type,int_from=nint_ids
        batch=batch[valid_ids]
        int_to=int_to[valid_ids]
        int_type=int_type[valid_ids]
        int_from=int_from[valid_ids]
        likelihoods=likelihoods[valid_ids]
        return batch,int_to,int_type,int_from,likelihoods
    
    def update_pl_adjs_with_nint_actions(self,batch,int_to,int_type,int_from,likelihoods,focus_lgroups,focus_lgroups_mask,p_groups,p_groups_mask):
        for ix,bid in enumerate(batch):
            if bid!=0:
                n_p_atoms=torch.sum(p_groups_mask[ix,int_to[ix]]).long()
                
                #print('int_to,bid',int_to,bid)
                
                int_p_atoms=p_groups[ix,int_to[ix]][:n_p_atoms]
                
                n_l_atoms=torch.sum(focus_lgroups_mask[ix]).long()
                
                int_l_atoms=focus_lgroups[ix,:n_l_atoms]-FGP.max_patoms

                int_types=int_type[ix]+5
                
                p_grids,l_grids=torch.meshgrid(int_p_atoms, int_l_atoms, indexing='ij')
                
                self.pl_adjs[bid,p_grids,l_grids,int_types]=1
                
                self.pl_adjs[bid]=self.pl_adjs[bid]*self.p_atom_int_masks[bid].unsqueeze(-1).unsqueeze(-1)
                
        self.nint_likelihoods[batch]=likelihoods
        return

    def update_l_graphs_with_conf_actions(self,ids,c_coords):
        self.l_coords[ids]=c_coords[:,FGP.max_patoms:]
        self.l_dmasks[ids]=self.l_gmasks[ids]
        return 
    
    def reset_zero_indices(self):
        self.l_stop_mask[0]=1
        self.r_stop_mask[0]=1
        self.__init_l_graph_zeros_indices()
        self.__init_r_graph_zeros_indices()
        self.l_likelihoods[0]=0
        self.r_likelihoods[0]=0
        self.nadd_likelihoods[0]=0
        self.nconn_likelihoods[0]=0
        self.nint_likelihoods[0]=0
        return 

    def sample_actions(self,apds,temp=1.0):
        apds=torch.exp(torch.log(apds)/temp)
        action_probs=torch.distributions.Multinomial(1,probs=apds)
        action=action_probs.sample()
        return action

    def compute_likelihoods(self,apds,action):
        likelihoods=torch.log(apds[action==1])
        return likelihoods

    def pred_nadd_apd(self,complex_nodes,complex_edges,complex_coords,complex_masks,nadd_constrain_masks=None):
        softmax=torch.nn.Softmax(dim=1)
        nadd_pred=self.node_add_model(complex_nodes,complex_edges,complex_coords,complex_masks)
        if nadd_constrain_masks is None:
            nadd_constrain_masks=torch.ones_like(nadd_pred)
        nadd_pred=softmax(nadd_pred)*nadd_constrain_masks.to(nadd_pred)
        return nadd_pred
    
    def split_nadd_action(self,actions):
        f_nadd=actions[:,:-1]
        f_term=actions[:,-1]
        add_idc=torch.nonzero(f_nadd,as_tuple=True)
        term_idc=torch.nonzero(f_term,as_tuple=True)
        return add_idc,term_idc
    
    def nadd_step(self,complex_nodes,complex_edges,complex_coords,complex_masks,nadd_constrain_masks=None,temp=1.0):
        nadd_apd=self.pred_nadd_apd(complex_nodes,complex_edges,complex_coords,complex_masks,nadd_constrain_masks)
        print (nadd_apd[1])
        nadd_action=self.sample_actions(nadd_apd,temp)
        add_idc,term_idc=self.split_nadd_action(nadd_action)
        likelihoods=self.compute_likelihoods(nadd_apd,nadd_action)
        return add_idc,term_idc,likelihoods

    def pred_rgen_apd(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,f_t,ring_masks,rgen_constrain_masks=None):
        softmax=torch.nn.Softmax(dim=1)
        rgen_pred=self.ring_gen_model(complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,f_t,ring_masks)
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

    def rgen_step(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,f_t,ring_masks,ring_atomnums,rgen_constrain_masks=None,temp=1.0):
        rgen_apd=self.pred_rgen_apd(complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,f_t,ring_masks,rgen_constrain_masks)
        rgen_action=self.sample_actions(rgen_apd,temp)
        likelihoods=self.compute_likelihoods(rgen_apd,rgen_action)
        add_idc,conn_idc,term_idc=self.split_rgen_action(rgen_action)
        # add bond_from to add_ids to judge the new added atom have beyond the maximum atoms
        r_add_froms=ring_atomnums[add_idc[0]]
        add_idc=(*add_idc,r_add_froms)
        # add bond_from to add_ids to judge the new added atom have beyond the maximum atoms
        r_conn_froms=ring_atomnums[conn_idc[0]]-1
        conn_idc=(*conn_idc,r_conn_froms)
        return add_idc,conn_idc,term_idc,likelihoods
    
    def invalid_rgen_actions(self,add_idc,conn_idc,term_idc,ring_edges,ring_atomnum,ring_maxatoms):
        
        #ring_maxatoms=torch.where(ring_atomnum>rw_ring_maxatoms,ring_atomnum,ft_ring_maxatoms).long()         
        # get invalid indices for when adding a new node to a non-empty graph
        empty_graphs=torch.nonzero(ring_atomnum[add_idc[0]]==0) # add atom connections to empty graphs
        invalid_add_to_non_empty=torch.nonzero(add_idc[1]>=ring_atomnum[add_idc[0]]) # add atom connections to unexist atoms
        combined=torch.cat((invalid_add_to_non_empty,empty_graphs),dim=0).squeeze(1)
        uniques,counts=combined.unique(return_counts=True)
        invalid_add_to_non_empty_ids=uniques[counts==1].unsqueeze(dim=1) 
        
        # get invalid indices for when adding a new node to an empty graph
        invalid_add_to_empty = torch.nonzero(add_idc[1]!=ring_atomnum[add_idc[0]])
        combined=torch.cat((invalid_add_to_empty,empty_graphs),dim=0).squeeze(1)
        uniques,counts=combined.unique(return_counts=True)
        invalid_add_to_empty_ids=uniques[counts>1].unsqueeze(dim=1)
        
        # get invalid indices for when connecting a node to an unpossible node which out of max ring size
        invalid_add_to_impossible_ids=torch.nonzero(add_idc[5]>=ring_maxatoms[add_idc[0]])
        # get invalid indices for when "connecting" a node in a graph with zero nodes
        invalid_conn_to_empty_ids=torch.nonzero(ring_atomnum[conn_idc[0]]==0)
        # get invalid indices for when connecting a node to nonexisting node
        invalid_conn_ids=torch.nonzero(conn_idc[1]>=ring_atomnum[conn_idc[0]])
        # get invalid indices for when connecting a node to itself
        invalid_conn_self_ids=torch.nonzero(conn_idc[1]==conn_idc[3])
        # get invalid indices for when attemting to add multiple edges
        invalid_conn_multi_ids=torch.nonzero(torch.sum(ring_edges,dim=-1)[conn_idc[0].long(),conn_idc[1].long(),conn_idc[-1].long()]==1)
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
    
    def pred_nconn_apd(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,focused_ids,ring_masks,nconn_constrain_masks=None):
        softmax=torch.nn.Softmax(dim=1)
        #print ('focused_ids',focused_ids,focused_ids.shape)
        conn_pred=self.node_conn_model(complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,focused_ids,ring_masks)
        if nconn_constrain_masks is None:
            nconn_constrain_masks=torch.ones_like(conn_pred)
        conn_pred=softmax(conn_pred)*nconn_constrain_masks.to(conn_pred) 
        return conn_pred
    
    def split_nconn_action(self,action):
        conn_shape=(action.shape[0],*FGP.leaf_conn_dim)
        f_conn=action.view(conn_shape)
        conn_idc=torch.nonzero(f_conn,as_tuple=True)
        return conn_idc
    
    def nconn_step(self,complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,focused_ids,ring_masks,mol_atomnums,nconn_constrain_masks=None,temp=1.0):
        
        conn_pred=self.pred_nconn_apd(complex_nodes,complex_edges,complex_coords,complex_masks,ring_nodes,ring_edges,focused_ids,ring_masks,nconn_constrain_masks)
        conn_action=self.sample_actions(conn_pred,temp)
        likelihoods=self.compute_likelihoods(conn_pred,conn_action)
        
        conn_idc=self.split_nconn_action(conn_action)
        conn_from=mol_atomnums[conn_idc[0]]+focused_ids[conn_idc[0]].view(-1)
        conn_idc=(*conn_idc,conn_from)
        
        return conn_idc,likelihoods
    
    def invalid_nconn_actions(self,conn_idc,mol_atomnum):
        empty_graphs=torch.nonzero(mol_atomnum[conn_idc[0]]==0)
        invalid_conn_ids=torch.nonzero(conn_idc[1]>=mol_atomnum[conn_idc[0]])
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
    
    def sample_mols(self,MG,nmols_per_complex=10,savepath='./sample_mols',debug=True):
        sampled_mols=[]
        sampled_smis=[]
        sampled_valids=[]
        nmols_per_complex=math.ceil(nmols_per_complex/(FGP.batchsize-1))*FGP.batchsize
        if self.local_rank is not None:
            sample_set=MG_Dataset([MG]*nmols_per_complex,name='complex',mode='pocket_only')
            sampler=torch.utils.data.distributed.DistributedSampler(sample_set)
            loader=DataLoader(sample_set,batch_size=FGP.batchsize,shuffle=False,num_workers=FGP.n_workers,sampler=sampler)
        else:
            sample_set=MG_Dataset([MG]*nmols_per_complex,name='complex',mode='pocket_only')
            loader=DataLoader(Train_Dataset,batch_size=FGP.batchsize,shuffle=False,num_workers=FGP.n_workers)

        sample_bar=enumerate(loader)
        
        MG_cp=copy.deepcopy(MG)
        Mass_center=MG_cp.get_pocket_mass_center()
        MG_cp.p_coords=MG_cp.p_coords-Mass_center
        MG_cp.l_coords=MG_cp.l_coords-Mass_center
        
        ref_ligmol=MG_cp.Trans_Ligand_to_Mol()
        ref_pocket=MG_cp.Trans_Pocket_to_Mol()
        

        os.system(f'mkdir -p {savepath}')
        molsupp=Chem.SDWriter(f'{savepath}/ref_lig.sdf')
        molsupp.write(ref_ligmol)
        molsupp.close()
        MolToXYZ(ref_ligmol,f'{savepath}/ref_lig.xyz')
        MolToXYZ(ref_pocket,f'{savepath}/ref_pocket.xyz')
        print ('Here')
        for bid,Datas in sample_bar: 
               
            self.sample_prepare()
            self.init_c_graphs(Datas)
            self.init_l_graphs(mode='all')
            # 0 index graph is always unused to avoid GGNN fails for empty graphs in the total batch.
            # thus it always needs to initialize the graph with only one node, its the same in ring generations.
            while torch.sum(self.l_stop_mask[1:])>0:
                self.init_nadd(mode='all')
                self.init_r_graphs(mode='all')
                self.init_nconn(mode='all')
                self.init_nint(mode='all')
                
                self.reset_zero_indices()
                all_ids=self.__to_device(torch.arange(FGP.batchsize).long())
                # initialize the 0 indexed graph with only one node
                # find unstoped complex graphs
                l_unstop_ids=torch.where(self.l_stop_mask==1)[0]
                c_feats,c_adjs_mat,c_coords,c_masks=self.select_c_graphs(l_unstop_ids,mask_mode='normal',output_mode='simple')
                # sample nadd actions for unstoped complex graphs
                l_add_ids,l_term_ids,nadd_likelihoods=self.nadd_step(c_feats,c_adjs_mat,c_coords,c_masks)
                self.l_stop_mask[l_unstop_ids[l_term_ids]]=0
                self.nadd_likelihoods[l_unstop_ids]=nadd_likelihoods
                # trans indices of nadd actions to real complex graph indices
                nadd_mol_ids=l_unstop_ids[l_add_ids[0]]
                nadd_type_ids=l_add_ids[1]
                # update single atom groups first
                self.r_stop_mask=self.l_stop_mask.clone().detach()    
                single_node_ids=torch.where(nadd_type_ids<len(FGP.atom_types))[0]            
                self.update_single_atom_r_graphs(nadd_mol_ids,nadd_type_ids,single_node_ids)
                self.r_stop_mask[nadd_mol_ids[single_node_ids]]=0 
                
                self.r_ftypes[nadd_mol_ids]=self.gid_to_gfeats[nadd_type_ids]
                self.update_r_maxatoms()
                #start fragment generation with rgen model
                self.record_nadd(path=f'{savepath}/{bid}')
                
                while torch.sum(self.r_stop_mask[1:])>0:
                    self.reset_zero_indices()
                    # find unstoped ring graphs
                    r_unstop_ids=torch.where(self.r_stop_mask==1)[0]

                    # get unstoped ring graphs and complex graphs
                    c_feats,c_adjs_mat,c_coords,c_masks=self.select_c_graphs(r_unstop_ids,mask_mode='normal',output_mode='simple')
                    
                    r_feats,r_adjs_mat,r_atomnums,r_atoms,r_ftypes,r_masks,r_round,r_maxatoms,r_focused_ids=self.select_r_graphs(r_unstop_ids)
                    # sample rgen actions
                    r_add_ids,r_conn_ids,r_term_ids,r_likelihoods=\
                                            self.rgen_step(c_feats,c_adjs_mat,c_coords,c_masks,
                                                                            r_feats,r_adjs_mat,r_ftypes,r_masks,r_atomnums)

                    # update the stop mask of ring graphs
                    r_invalid_ids,r_full_ids=self.invalid_rgen_actions(r_add_ids,r_conn_ids,r_term_ids,r_adjs_mat,r_atomnums,r_maxatoms)

                    
                    self.r_stop_mask[r_unstop_ids[r_term_ids]]=0
                    self.r_stop_mask[r_unstop_ids[r_full_ids]]=0
                    self.r_likelihoods[r_unstop_ids[r_term_ids],self.r_rounds[r_unstop_ids[r_term_ids]]]=r_likelihoods[r_term_ids]
                    
                    # full ids are invalid actions, should not be updated by RL, thus, we didn't collect its likelihoods
                    self.init_r_graphs(ids=r_unstop_ids[r_invalid_ids],mode='failed')
                    
                    r_valid_mask=self.r_stop_mask.clone().detach()
                    r_valid_mask[r_unstop_ids[r_invalid_ids]]=0
                    r_add_valid_ids=torch.nonzero(r_valid_mask[r_unstop_ids[r_add_ids[0]]]>0)
                    r_conn_valid_ids=torch.nonzero(r_valid_mask[r_unstop_ids[r_conn_ids[0]]]>0)
                    
                    batch,bond_to,atom_type,charge,bond_type,bond_from,likelihoods\
                        =self.select_valid_r_add_actions(r_add_ids,r_add_valid_ids,r_likelihoods)
                    batch=r_unstop_ids[batch]
                    self.update_r_graphs_with_add_actions(batch,bond_to,atom_type,charge,bond_type,bond_from,likelihoods)
                    
                    batch,bond_to,bond_type,bond_from,likelihoods=self.select_valid_r_conn_actions(r_conn_ids,r_conn_valid_ids,r_likelihoods)
                    batch=r_unstop_ids[batch]
                    self.update_r_graphs_with_conn_actions(batch,bond_to,bond_type,bond_from,likelihoods)
                    #print (self.r_feats[1])

                self.record_rgen(path=f'{savepath}/{bid}')
                # update the complex graphs with the generated fragments
                self.reset_zero_indices()
                l_full_ids=torch.where(self.l_atomnums+self.r_atomnums>FGP.max_latoms)
                self.l_stop_mask[l_full_ids]=0
                self.nconn_stop_mask=self.l_stop_mask.clone().detach()
                nconn_unstop_ids=torch.where(self.nconn_stop_mask>0)[0]
                
                c_feats,c_adjs_mat,c_coords,c_masks\
                    =self.select_c_graphs(nconn_unstop_ids,mask_mode='normal',output_mode='simple')
                    
                r_feats,r_adjs_mat,r_atomnums,r_atoms,r_ftypes,r_masks,r_rounds,r_maxatoms,r_focused_ids=self.select_r_graphs(nconn_unstop_ids)
                
                l_atomnums=self.l_atomnums[nconn_unstop_ids]
                nconn_ids,nconn_likelihoods=\
                    self.nconn_step(c_feats,c_adjs_mat,c_coords,c_masks,r_feats,r_adjs_mat,r_focused_ids,r_masks,l_atomnums)
                
                # select valid node connection actions
                nconn_invalid_ids=self.invalid_nconn_actions(nconn_ids,l_atomnums)
                
                nconn_valid_mask=self.nconn_stop_mask.clone().detach()
                nconn_valid_mask[nconn_unstop_ids[nconn_invalid_ids]]=0
                
                nconn_valid_ids=torch.nonzero(nconn_valid_mask[nconn_unstop_ids[nconn_ids[0]]]>0).view(-1)

                # select valid node connection actions
                batch,bond_to,bond_type,bond_from,likelihoods=self.select_valid_nconn_actions(nconn_ids,nconn_valid_ids,nconn_likelihoods)
                batch=nconn_unstop_ids[batch]
                
                self.update_l_graphs_with_nconn_actions(batch, bond_to, bond_type, bond_from, likelihoods)
                
                # update the l graphs with the generated fragments
                l_update_ids=torch.where(self.l_stop_mask>0)[0]
                started_l_ids=torch.where(self.l_rounds[l_update_ids]==0)[0]
                started_l_ids=l_update_ids[started_l_ids]
                
                updated_ids=torch.unique(torch.cat((started_l_ids,batch)))
                self.update_l_graphs_with_rgraphs(updated_ids)
                self.record_nconn(path=f'{savepath}/{bid}')
                
                self.nint_stop_mask=self.__to_device(torch.zeros(FGP.batchsize)).long()
                self.nint_stop_mask[updated_ids]=1
                
                counter=0
                while torch.sum(self.nint_stop_mask[1:])>0:
                    self.reset_zero_indices()
                    self.nint_stop_mask[0]=1
                    nint_unstop_ids=torch.where(self.nint_stop_mask>0)[0]
                    r_feats,r_adjs_mat,r_atomnums,r_atoms,r_ftypes,r_masks,r_round,r_maxatoms,r_focused_ids=self.select_r_graphs(nint_unstop_ids)
                    
                    cg_feats,cg_atoms,cg_adjs_mat,cg_coords,cg_masks,\
                        p_groups,p_group_masks,n_pgroups,p_group_int_masks,p_atom_int_masks=self.select_c_graphs(nint_unstop_ids,mask_mode='graph',output_mode='middle')
                    #print (cg_adjs_mat[1,FGP.max_patoms:FGP.max_patoms+15,0])
                    cd_feats,cd_atoms,cd_adjs_mat,cd_coords,cd_masks,\
                        p_groups,p_group_masks,n_pgroups,p_group_int_masks,p_atom_int_masks=self.select_c_graphs(nint_unstop_ids,mask_mode='coords',output_mode='middle')

                    focus_ftypes=r_ftypes
                    focus_lgroups=self.l_groups[nint_unstop_ids,self.l_rounds[nint_unstop_ids]-1]+FGP.max_patoms
                    focus_lgroup_masks=self.l_groups_masks[nint_unstop_ids,self.l_rounds[nint_unstop_ids]-1]
                    
                    focus_lgroup_ids=self.l_rounds[nint_unstop_ids]-1
                    print ('p_group_int_masks',p_group_int_masks.shape)
                    nint_ids,nint_term_ids,nint_likelihoods=self.nint_step(cg_feats,cg_adjs_mat,cg_masks,
                                                                                      cd_feats,cd_adjs_mat,cd_coords,cd_masks,
                                                                                      p_groups,p_group_masks,p_group_int_masks,
                                                                                      focus_lgroups,focus_lgroup_masks,focus_ftypes,focus_lgroup_ids)
                    
                    self.nint_stop_mask[nint_unstop_ids[nint_term_ids]]=0
                    self.nint_likelihoods[nint_unstop_ids[nint_term_ids]]=nint_likelihoods[nint_term_ids]
                    self.nint_stop_mask[nint_unstop_ids[nint_term_ids]]=0
                    
                    nint_invalid_ids=self.invalid_nint_actions(nint_ids,n_pgroups)
                    
                    self.nint_stop_mask[nint_unstop_ids[nint_invalid_ids]]=0
                    
                    nint_valid_masks=self.nint_stop_mask.clone().detach()
                    nint_valid_masks[nint_unstop_ids[nint_invalid_ids]]=0
                    nint_valid_ids=torch.nonzero(nint_valid_masks[nint_unstop_ids[nint_ids[0]]]>0).view(-1)
                    
                    batch,int_to,int_type,int_from,likelihoods=self.select_valid_nint_actions(nint_ids,nint_valid_ids,nint_likelihoods)
                    batch=nint_unstop_ids[batch]
                    
                    self.update_pl_adjs_with_nint_actions(batch,int_to,int_type,int_from,likelihoods,focus_lgroups,focus_lgroup_masks,p_groups,p_group_masks)
                    counter+=1

                # update the coords of the complex graphs
                coords_unstop_ids=updated_ids.clone().detach()
                if len(coords_unstop_ids)==0:
                    break
                #if len(coords_unstop_ids)>0:
                if True:
                    c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,c_fix_masks,c_flexible_masks,\
                        p_groups,p_groups_masks,n_pgroups,p_groups_int_masks,p_groups_atom_int_masks,p_atom_int_masks,c_pocket_labels,c_ligand_labels\
                            =self.select_c_graphs(coords_unstop_ids,mask_mode='graphs',output_mode='all')
                    b,n,_,d=c_adjs_mat.shape
                    indices=torch.arange(n).to(self.device)
                    print ('c_adjs_mat diagonals',c_adjs_mat[:,indices,indices].shape,torch.where(torch.sum(c_adjs_mat[:,indices,indices],dim=-1)))
                    
                    c_adjs_mat[0,FGP.max_patoms:,FGP.max_patoms:]=0
                    
                    samples,samples_diff,samples_diff_before=self.inpaint_coords_batch(\
                                    c_feats,c_atoms,c_adjs_mat,c_coords,c_masks,\
                                    c_fix_masks,c_flexible_masks,c_pocket_labels,c_ligand_labels)
                    
                    self.update_l_graphs_with_conf_actions(coords_unstop_ids,samples)
                    
                    self.record_conf(path=f'{savepath}/{bid}')
                    
            mols,smis,valid_ids=self.__l_graphs_to_mol(with_coords=True)
            sampled_mols+=mols
            sampled_smis+=smis
            sampled_valids+=valid_ids
            self.record_conf(path=f'{savepath}')
                
        return mols,smis,valid_ids
                    
    def __node_feats_to_atom(self,node_feat):
        non_zero_idc=torch.nonzero(node_feat>0)
        atom_idx=non_zero_idc[0]
        atom_type=FGP.atom_types_for_feats[atom_idx]
        new_atom=Chem.Atom(atom_type)
        fc_idx=non_zero_idc[1]-len(FGP.atom_types_for_feats)
        formal_charge=FGP.formal_charge_types[fc_idx]
        new_atom.SetFormalCharge(formal_charge)
        return new_atom
                
    def __l_graphs_to_mol(self,with_coords=False):
        if with_coords: 
            mols,smis,valid_ids=self.graph_to_mols(self.l_feats[1:],self.l_adjs_mat[1:],self.l_atomnums[1:],self.l_coords[1:])
        else:
            mols,smis,valid_ids=self.graph_to_mols(self.l_feats[1:],self.l_adjs_mat[1:],self.l_atomnums[1:])
        return mols,smis,valid_ids

    def __r_graphs_to_mol(self):
        mols,smis,valid_ids=self.graph_to_mols(self.r_feats[1:],self.r_adjs_mat[1:],self.r_atomnums[1:])
        return mols,smis,valid_ids
    
    def graph_to_mols(self,feats,adjs_mat,atomnums,coords=None): 
        mols=[]
        smis=[]
        batchsize=feats.shape[0]
        graphsize=feats.shape[1]
        valid_ids=[0]*batchsize
        for i in range(batchsize):
            mol=Chem.RWMol()
            node_to_idx={}
            print ('atomnums',atomnums[i])
            for j in range(atomnums[i]):
                atom=self.__node_feats_to_atom(feats[i,j])
                mol_idx=mol.AddAtom(atom)
                node_to_idx[j]=mol_idx
            print ('node_to_idx',node_to_idx)
            edge_mask=self.__to_device(torch.triu(
                    torch.ones((graphsize, graphsize)),
                    diagonal=1
                    ).unsqueeze(-1)).long()
            edge_idc=torch.nonzero(edge_mask*adjs_mat[i])
            for idx1,idx2,bond_index in edge_idc:
                #try:
                    bond_type=FGP.bond_types[bond_index-1]
                    mol.AddBond(node_to_idx[idx1.item()],node_to_idx[idx2.item()],bond_type) 
                #except:
                #    pass
            try:
            #if True:
                mol=mol.GetMol()
                AllChem.Compute2DCoords(mol)
                #Chem.SanitizeMol(mol)
                smi=Chem.MolToSmiles(mol)
                if coords is not None:
                    mol=Change_mol_xyz(mol,coords[i].clone().detach().cpu().numpy()[:atomnums[i]])
                mols.append(mol)
                smis.append(smi)
                valid_ids[i]=1
                #print (mol,smi)
            except:
                mols.append(None)
                smis.append(None)  
            
        return mols,smis,valid_ids 

    def record_nadd(self,path=f'./samples/'): 
        l_mols,l_smis,l_valid_ids=self.__l_graphs_to_mol()
        for i in range(len(l_mols)):
            try:
                savepath=f"{path}-{i+1}/{self.l_rounds[i+1]}"
                if not os.path.exists(savepath):
                    os.system(f"mkdir -p {savepath}")
                with open(f"{savepath}/l_nadd.rftype",'w') as gf:
                    f=self.r_ftypes[i+1].long()
                    if f[0]>0:
                        descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}-HD{f[12]}-HA{f[13]}-Neg{f[14]}-Pos{f[15]}-Aro{f[16]}-Hyd{f[17]}-LHyd{f[18]}'
                    else:
                        if f[10]>1:
                            descriptor=f'R{f[0]}-C*-N*-O*-F*-P*-S*-Cl*-Br*-I*-Bes{f[10]}-AR*-'+\
                                f'HD{f[12]}-HA{f[13]}-Neg{f[14]}-Pos{f[15]}-Aro{f[16]}-Hyd{f[17]}-LHyd{f[18]}'                   
                        else:
                            descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR*-'+\
                            f'HD*-HA*-Neg*-Pos*-Aro*-Hyd*-LHyd*'
                    gf.write(descriptor+'\n')
                if l_mols[i] is not None:
                    with open(f"{savepath}/l_nadd.smi",'w') as f:
                        f.write(l_smis[i])
                    writer=Chem.SDWriter(f"{savepath}/l_nadd.sdf")
                    writer.write(l_mols[i])
                    writer.close()
                    try:
                        Draw.MolToFile(l_mols[i],f"{savepath}/l_nadd.png")
                    except:
                        pass
            except:
                print (f'Error in record nadd for {savepath}/l_nadd.png')
        return
        
    def record_rgen(self,path=f'./samples'):
        r_mols,r_smis,r_valid_ids=self.__r_graphs_to_mol()
        for i in range(len(r_mols)):
            if r_mols[i] is not None:
                try:
                    savepath=f"{path}-{i+1}/{self.l_rounds[i+1]}"
                    if not os.path.exists(savepath):
                        os.system(f"mkdir -p {savepath}")
                    with open(f"{savepath}/r_gen.smi",'w') as f:
                        f.write(r_smis[i])
                    writer=Chem.SDWriter(f"{savepath}/r_gen.sdf")
                    writer.write(r_mols[i])
                    writer.close()
                    try:
                        Draw.MolToFile(r_mols[i],f"{savepath}/r_gen.png")
                    except:
                        pass
                except:
                    print (f'Error in record rgen for {savepath}/r_gen.png')
        return
        
    def record_nconn(self,path=f'./samples'):
        l_mols,l_smis,l_valid_ids=self.__l_graphs_to_mol()
        for i in range(len(l_mols)):
            if l_mols[i] is not None:
                try:
                    savepath=f"{path}-{i+1}/{self.l_rounds[i+1]-1}"
                    os.system(f"mkdir -p {savepath}")
                    with open(f"{savepath}/l_nconn.smi",'w') as f:
                        f.write(l_smis[i])
                    writer=Chem.SDWriter(f"{savepath}/l_nconn.sdf")
                    writer.write(l_mols[i])
                    writer.close()
                    try:
                        Draw.MolToFile(l_mols[i],f"{savepath}/l_nconn.png")
                    except:
                        pass
                except:
                    print (f'Error in record nconn for {savepath}/l_nconn.png')
        return
        
    def record_conf(self,path=f'./samples'):
        l_mols,l_smis,l_valid_ids=self.__l_graphs_to_mol(with_coords=True)
        for i in range(len(l_mols)):
            if l_mols[i] is not None:
                try:
                    savepath=f"{path}-{i+1}/{self.l_rounds[i+1]-1}"
                    os.system(f"mkdir -p {savepath}")
                    with open(f"{savepath}/l_conf.smi",'w') as f:
                        f.write(l_smis[i])
                    writer=Chem.SDWriter(f"{savepath}/l_conf.sdf")
                    writer.write(l_mols[i])
                    writer.close()
                    
                    try:
                        Draw.MolToFile(l_mols[i],f"{savepath}/l_conf.png")
                    except:
                        pass
                except:
                    print (f'Error in record conf for {savepath}/l_conf.png')
        return
         
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
    
    def __init_l_graph_zeros_indices(self):
        dummy_af=torch.Tensor(onek_encoding_unk(6,FGP.atom_types_for_feats)+\
                onek_encoding_unk(0,FGP.formal_charge_types))
        self.l_feats[0]=self.__to_device(dummy_af).float()
        self.l_adjs_mat[0,0,0,0]=1
        self.l_atomnums[0]=1
        self.l_atoms[0,0]=6
        self.l_masks[0,0]=1
        return 
     
    def __init_r_graph_zeros_indices(self):
        dummy_af=torch.Tensor(onek_encoding_unk(6,FGP.atom_types_for_feats)+onek_encoding_unk(0,FGP.formal_charge_types)).float()
        self.r_feats[0]=self.__to_device(dummy_af)
        self.r_adjs_mat[0,0,0,0]=1
        self.r_atomnums[0]=1
        self.r_atoms[0,0]=6
        self.r_masks[0,0]=1
        self.r_ftypes[0]=0
        self.r_maxatoms[0]=1
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
