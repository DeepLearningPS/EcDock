# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np
from eccore import metrics
from eccore.losses import UnicoreLoss, register_loss
import collections


@register_loss("docking_pose_v2")
class DockingPosseV2Loss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.eos_idx = task.dictionary.eos()
        self.bos_idx = task.dictionary.bos()
        self.padding_idx = task.dictionary.pad()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        cross_distance_predict, holo_distance_predict, coord_predict, prmsd_predict = net_output[:4]
        
        print('cross_distance_predict[0]:', cross_distance_predict[0])
        print('holo_distance_predict[0]:', holo_distance_predict[0]) #nan
        print('distance_target[0]:', sample["target"]["distance_target"][0])
        

        ### distance loss
        distance_mask = sample["target"]["distance_target"].ne(0) # 0 is padding
        if self.args.dist_threshold > 0:
            distance_mask &= sample["target"]["distance_target"] < self.args.dist_threshold
        distance_predict = cross_distance_predict[distance_mask]
        distance_target =  sample["target"]["distance_target"][distance_mask]
        distance_loss = F.mse_loss(
            distance_predict.float(), 
            distance_target.float(), 
            reduction="mean")
        
        ### holo distance loss
        token_mask = sample["net_input"]["mol_src_tokens"].ne(self.padding_idx) & \
                     sample["net_input"]["mol_src_tokens"].ne(self.eos_idx) & \
                     sample["net_input"]["mol_src_tokens"].ne(self.bos_idx)
        holo_distance_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(1)
        holo_distance_predict = holo_distance_predict[holo_distance_mask]
        holo_distance_target =  sample["target"]["holo_distance_target"][holo_distance_mask]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict.float(), 
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
            )

        ### coord loss
        coord_target = sample["target"]["holo_coord"]
        coord_mask = coord_target.ne(0)  # 0 is padding
        coord_loss = (((coord_predict - coord_target)**2).sum(dim=[1,2]) / coord_mask[:,:,0].sum(dim=-1)).sqrt().mean()

        ### prmsd loss
        tick = 0.25
        max_bins = 32 
        token_mask = coord_mask[:,:,0]
        prmsd_target = ((coord_predict - coord_target)**2 * coord_mask).sum(dim=-1).sqrt()
        prmsd_target = (prmsd_target / tick).long()
        prmsd_target[prmsd_target >= (max_bins - 1)] = max_bins - 1
        prmsd_target[prmsd_target < 0] = 0
        prmsd_logit = F.softmax(prmsd_predict.float(), dim=-1)   # BS, N, MAX_BINS
        prmsd_predict = F.log_softmax(prmsd_predict.float(), dim=-1)   # BS, N, MAX_BINS
        prmsd_loss = F.nll_loss(
            prmsd_predict[token_mask],
            prmsd_target[token_mask],
            reduction="mean",
        )

        loss = distance_loss + holo_distance_loss + coord_loss + + prmsd_loss*0.1
        print('loss:', loss.item(), 'distance_loss:', distance_loss.item(), 'holo_distance_loss:', holo_distance_loss.item(), 'coord_loss:', coord_loss.item(), 'prmsd_loss:', prmsd_loss.item())
        weight = torch.arange(max_bins,).type_as(prmsd_logit).unsqueeze(0) + tick / 2
        prmsd_score = (prmsd_logit * weight).sum(dim=-1).mean(dim=-1)

        sample_size = sample["target"]["holo_coord"].size(0)
        logging_output = {
            "loss": loss.data,
            "cross_distance_loss": distance_loss.data,
            "distance_loss": holo_distance_loss.data,
            "coord_loss": coord_loss.data,
            "prmsd_loss": prmsd_loss.data,
            "prmsd_score": prmsd_score.data,
            "bsz": sample_size,
            "sample_size": 1,
            "coord_predict": coord_predict.data,   # last iteration
            "coord_target": sample["target"]["holo_coord"].data,
        }
        if not self.training:
            logging_output["smi_name"] = sample["smi_name"]
            logging_output["pocket_name"] = sample["pocket_name"]
            logging_output["coord_predict"] = coord_predict.data.detach().cpu()
            logging_output["prmsd_score"] = prmsd_score.data.detach().cpu()
            logging_output["atoms"] = sample["net_input"]["mol_src_tokens"].data.detach().cpu()
            logging_output["pocket_atoms"] = sample["net_input"]["pocket_src_tokens"].data.detach().cpu()
            logging_output["coordinates"] = sample["net_input"]["mol_src_coord"].data.detach().cpu()
            logging_output["holo_coordinates"] = sample["target"]["holo_coord"].data.detach().cpu()
            logging_output["pocket_coordinates"] = sample["net_input"]["pocket_src_coord"].data.detach().cpu()
            logging_output["holo_center_coordinates"] = sample["holo_center_coordinates"].data.detach().cpu()
            
        #print('coord_predict.data.detach().cpu()[:2]:', coord_predict.data.detach().cpu()[:2])
        print('coord_predict.data.detach().cpu().shape:', coord_predict.data.detach().cpu().shape)
        
        print('coord_predict.data.detach().cpu()[:2].sum():', coord_predict.data.detach().cpu()[:2].sum())
        #print('holo_coordinates[:2]:', sample["net_input"]["pocket_src_coord"].data.detach().cpu()[:2])
        
        return loss, sample_size, logging_output



    def forward_sample_copy(self, model, sample, reduce=True):
    #try:
        ## ，0，,，，，
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"]) #holo_coord
        cross_distance_predict, holo_distance_predict, coord_predict, prmsd_predict = net_output[:4]
        ##print('cross_distance_predict, holo_distance_predict, coord_predict, prmsd_predict:', 
        #        cross_distance_predict.shape, holo_distance_predict.shape, coord_predict.shape, prmsd_predict.shape)
        #torch.Size([1, 16, 40]) torch.Size([1, 16, 16]) torch.Size([1, 16, 3]) torch.Size([1, 16, 32]
        
        
        print('cross_distance_predict[0]:', cross_distance_predict[0])
        print('holo_distance_predict[0]:', holo_distance_predict[0]) #nan
        print('distance_target[0]:', sample["target"]["distance_target"][0])
        
        
        '''
        .ne(0)  PyTorch ，。， True，
         False。，。
        '''
        ### distance loss, ，0，
        ## ，0，
        distance_mask = sample["target"]["distance_target"].ne(0) # 0 is padding， ，，1，，1
        #if self.args.dist_threshold > 0: #8
            #distance_mask &= sample["target"]["distance_target"] < self.args.dist_threshold #，，dist_threshold=8ai
        distance_predict = cross_distance_predict[distance_mask]

        #print('distance_mask:', distance_mask.shape) # torch.Size([2, 16, 136])
        #print('cross_distance_predict:', cross_distance_predict.shape) #torch.Size([2, 16, 136])
        #print('cross_distance_predict[distance_mask]:', cross_distance_predict[distance_mask].shape) #torch.Size([3328]) = 2*13*128, 13，128
        distance_target =  sample["target"]["distance_target"][distance_mask] #，shape = [2, n*m]
        #mse：，
        distance_loss = F.mse_loss(
            distance_predict.float(), 
            distance_target.float(), 
            reduction="mean")
    

        #distance_predict，。，，，40，
        #


        
        ### holo distance loss #holo？
        token_mask = sample["net_input"]["mol_src_tokens"].ne(self.padding_idx) & \
                    sample["net_input"]["mol_src_tokens"].ne(self.eos_idx) & \
                    sample["net_input"]["mol_src_tokens"].ne(self.bos_idx)
        holo_distance_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(1)
        holo_distance_predict = holo_distance_predict[holo_distance_mask]
        holo_distance_target =  sample["target"]["holo_distance_target"][holo_distance_mask]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict.float(), 
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
            )
        '''
        Smooth L1 Loss（Huber Loss），。
        （MSE）（MAE），
        Smooth L1 Loss：

        ，MSE，。
        ，MAE，。
        
        '''

        ### coord loss
        coord_target = sample["target"]["holo_coord"]
        coord_mask = coord_target.ne(0)  # 0 is padding
        #print('pocket_name:', sample["pocket_name"])
        #print('coord_mask:', coord_mask.shape) #torch.Size([2, 16, 3])
        #print('coord_target:', coord_target.shape) #torch.Size([2, 16, 3])
        #print('coord_predict:', coord_predict.shape) #torch.Size([2, 16, 3])
        #print('coord_target[coord_mask]:', coord_target[coord_mask].shape) #torch.Size([78]) # 78 =  2*13*3
        #print('coord_predict[coord_mask]:', coord_predict[coord_mask].shape) #torch.Size([78])

        #
        coord_loss = (((coord_predict - coord_target)**2).sum(dim=[1,2]) / coord_mask[:,:,0].sum(dim=-1)).sqrt().mean()

        ### prmsd loss
        tick = 0.25
        max_bins = 32 
        token_mask = coord_mask[:,:,0]
        prmsd_target = ((coord_predict - coord_target)**2 * coord_mask).sum(dim=-1).sqrt()
        prmsd_target = (prmsd_target / tick).long()
        prmsd_target[prmsd_target >= (max_bins - 1)] = max_bins - 1
        prmsd_target[prmsd_target < 0] = 0
        prmsd_logit = F.softmax(prmsd_predict.float(), dim=-1)   # BS, N, MAX_BINS
        prmsd_predict = F.log_softmax(prmsd_predict.float(), dim=-1)   # BS, N, MAX_BINS
        prmsd_loss = F.nll_loss(
            prmsd_predict[token_mask],
            prmsd_target[token_mask],
            reduction="mean",
        )

        loss = distance_loss + holo_distance_loss + coord_loss + prmsd_loss*0.1
        print('loss:', loss.item(), 'distance_loss:', distance_loss.item(), 'holo_distance_loss:', holo_distance_loss.item(), 'coord_loss:', coord_loss.item(), 'prmsd_loss:', prmsd_loss.item())

        weight = torch.arange(max_bins,).type_as(prmsd_logit).unsqueeze(0) + tick / 2
        prmsd_score = (prmsd_logit * weight).sum(dim=-1).mean(dim=-1)

        sample_size = sample["target"]["holo_coord"].size(0)
        logging_output = {
            "loss": loss.data,
            "cross_distance_loss": distance_loss.data,
            "distance_loss": holo_distance_loss.data,
            "coord_loss": coord_loss.data,
            "prmsd_loss": prmsd_loss.data,
            "prmsd_score": prmsd_score.data,
            "bsz": sample_size,
            "sample_size": 1,
            "coord_predict": coord_predict.data,   # last iteration
            "coord_target": sample["target"]["holo_coord"].data,
        }

        #，
        #xyz0，bool，，0，
        #ligand_shape  = coord_target[coord_mask].view(coord_target.size(0), -1, coord_target.size(2)).shape
        #，xyz0，TRUE
        coord_mask = coord_target.ne(0).any(dim=-1)  #  (b, n)
        ligand_shape = coord_target[coord_mask].view(coord_target.size(0), -1, coord_target.size(2)).shape
        
        #，，? ，，
        holo_coord_pocket = sample["target"]["holo_coord_pocket"]
        #print('tyep(holo_coord_pocket):', type(holo_coord_pocket))
        coord_pocket_mask = holo_coord_pocket.ne(0).any(dim=-1)
        
        #exit()
        #，holo_coord_pocket0
        #40rdkit，，，，，

        #print('holo_coord_pocket.shape:', holo_coord_pocket.shape)
        #print('coord_pocket_mask.shape:', coord_pocket_mask.shape)
        #print('coord_pocket_mask Not 0 num:', torch.sum(coord_pocket_mask))
        

            
        #7PK0，256，，040*256*3 = 30720，，30719，，
        #error: shape '[40, -1, 3]' is invalid for input of size 30719， holo_coord_pocket0
        
        

        assert torch.sum(coord_pocket_mask) == len(holo_coord_pocket[coord_pocket_mask])

        #try:
        protein_shape = holo_coord_pocket[coord_pocket_mask].view(holo_coord_pocket.size(0), -1, holo_coord_pocket.size(2)).shape
        #except Exception as e:
            ##print(f"Error occurred while calculating protein shape: {e}")
            ##print('-----------------------------------------------')
            ##print('pocket_name:', sample['pocket_name'])
            ##print('coord_pocket_mask:', coord_pocket_mask.shape)
            ##print('holo_coord_pocket:', holo_coord_pocket.shape)
            ##print('holo_coord_pocket[coord_pocket_mask]:', holo_coord_pocket[coord_pocket_mask].shape)

            #raise Exception('protein error')

        cross_distance_target_shape = [ligand_shape[0], ligand_shape[1], protein_shape[1]]
        #print('ligand_shape:', ligand_shape)
        #print('protein_shape:', protein_shape)
        #print('cross_distance_target_shape:', cross_distance_target_shape)


        if not self.training:
            logging_output["smi_name"] = sample["smi_name"]
            logging_output["pocket_name"] = sample["pocket_name"]
            logging_output["coord_predict"] = coord_predict[coord_mask].view(ligand_shape).data.detach().cpu()
            logging_output["prmsd_score"] = prmsd_score.data.detach().cpu()
            logging_output["atoms"] = sample["net_input"]["mol_src_tokens"].data.detach().cpu()
            logging_output["pocket_atoms"] = sample["net_input"]["pocket_src_tokens"].data.detach().cpu()
            logging_output["coordinates"] = sample["net_input"]["mol_src_coord"].data.detach().cpu()

            #3, 0
            logging_output["holo_coordinates"] = sample["target"]["holo_coord"][coord_mask].view(ligand_shape).data.detach().cpu()
            logging_output["holo_pocket_coordinates"] = sample["target"]["holo_coord_pocket"][coord_pocket_mask].view(protein_shape).data.detach().cpu()
            #logging_output["holo_pocket_coordinates"] = sample["holo_coord_pocket"].data.detach().cpu()
            logging_output["pocket_coordinates"] = sample["net_input"]["pocket_src_coord"][coord_pocket_mask].view(protein_shape).data.detach().cpu()
            logging_output["cross_distance"] = cross_distance_predict[distance_mask].view(cross_distance_target_shape).data.detach().cpu() #，，     

            #print('logging_output["coord_predict"]:', logging_output["coord_predict"].shape)
            #print('logging_output["holo_coordinates"]:', logging_output["holo_coordinates"].shape)
            #print('logging_output["pocket_coordinates"]:', logging_output["pocket_coordinates"].shape)
            #print('logging_output["cross_distance"]:', logging_output["cross_distance"].shape)
            #print('logging_output["holo_pocket_coordinates"]:', logging_output["holo_pocket_coordinates"].shape)
            #logging_output["coord_predict"]: torch.Size([40, 16, 3])
            #logging_output["holo_coordinates"]: torch.Size([40, 16, 3])
            #logging_output["pocket_coordinates"]: torch.Size([40, 136, 3])
            #logging_output["cross_distance"]: torch.Size([40, 16, 136])
            #logging_output["holo_pocket_coordinates"]: torch.Size([40, 136, 3]) #
            #raise Exception('stop')
            #raise Exception('stop')
            logging_output["holo_center_coordinates"] = sample["holo_center_coordinates"].data.detach().cpu()
            
            '''
            logging_output["coord_predict"]: torch.Size([1, 31, 3])
            logging_output["holo_coordinates"]: torch.Size([1, 31, 3])
            logging_output["pocket_coordinates"]: torch.Size([1, 256, 3])
            logging_output["cross_distance"]: torch.Size([1, 31, 256])
            logging_output["holo_pocket_coordinates"]: torch.Size([1, 256, 3])
            '''
            assert logging_output["cross_distance"].shape[1] == logging_output["coord_predict"].shape[1]
            assert logging_output["cross_distance"].shape[2] == logging_output["pocket_coordinates"].shape[1]
            assert logging_output["holo_coordinates"].shape[1] == logging_output["coord_predict"].shape[1]
            assert logging_output["holo_pocket_coordinates"].shape[1] == logging_output["pocket_coordinates"].shape[1]

            '''
            #print(len(sample["target"]["holo_coord_pocket"]))
            for i in range(len(sample["target"]["holo_coord_pocket"])):
                assert torch.allclose(sample["target"]["holo_coord_pocket"][i-1], sample["target"]["holo_coord_pocket"][i], rtol=0.01, atol=0.02)
            #raise Exception('test')
            '''
            
        #return loss, sample_size, (logging_output, sample['pocket_name'])
        return loss, sample_size, logging_output
    
    '''
    except Exception as e:
        #print('error:', e)
        ##print('error name:', sample['pocket_name'])
        #40rdkit，，，，，
        #raise Exception('error,stop')
        #return None, None, (None, sample['pocket_name'])
        return None, None, None
    '''
    
    def get_unmasked_distance_matrix(self, mol_padding_mask, pocket_padding_mask, cross_distance_predict):
        """
        ，
        
        :
            mol_padding_mask:  [batch_size, max_mol_len]
            pocket_padding_mask:  [batch_size, max_pocket_len]
            cross_distance_predict:  [batch_size, max_mol_len, max_pocket_len]
        
        :
            ()
        """
        batch_size = mol_padding_mask.size(0)
        unmasked_matrices = []
        
        for i in range(batch_size):
            # 
            mol_mask = mol_padding_mask[i].bool()  # [max_mol_len]
            pocket_mask = pocket_padding_mask[i].bool()  # [max_pocket_len]
            
            # 
            mol_len = mol_mask.sum().item()
            pocket_len = pocket_mask.sum().item()
            
            # 
            valid_mol_indices = mol_mask.nonzero().squeeze(-1)  # [mol_len]
            valid_pocket_indices = pocket_mask.nonzero().squeeze(-1)  # [pocket_len]
            
            # 
            valid_dist_matrix = cross_distance_predict[i][valid_mol_indices][:, valid_pocket_indices]  # [mol_len, pocket_len]
            
            # 2
            valid_dist_matrix = valid_dist_matrix[1:-1, 1:-1]
            
            unmasked_matrices.append(valid_dist_matrix)
        
        return torch.stack(unmasked_matrices, dim = 0)
    

    def get_unmasked_pos_emb_(self, mask, matrix):
        """
        ，
        
        :
            mol_padding_mask:  [batch_size, max_mol_len]
            pocket_padding_mask:  [batch_size, max_pocket_len]
            cross_distance_predict:  [batch_size, max_mol_len, max_pocket_len]
        
        :
            ()
        """
        batch_size = mask.size(0)
        #print('batch_size:', batch_size)
        unmasked_matrices = []
        
        for i in range(batch_size):
            #print('i:', i)
            # 
            mol_mask = mask[i].bool()  # [max_mol_len]
            
            # 
            mol_len = mol_mask.sum().item()
            
            # 
            valid_mol_indices = mol_mask.nonzero().squeeze(-1)  # [mol_len]
            
            # 
            valid_matrix = matrix[i][valid_mol_indices]  # [mol_len, pocket_len]
            
            # 2
            valid_dist_matrix = valid_matrix[1:-1,:]
            #print('valid_dist_matrix:', valid_dist_matrix.shape)
            
            unmasked_matrices.append(valid_dist_matrix)
        #print('torch.stack(unmasked_matrices, dim = 0):', torch.stack(unmasked_matrices, dim = 0).shape)
        return torch.stack(unmasked_matrices, dim = 0)
    
    def get_unmasked_2d(self, mask, matrix):
        """
        ，
        
        :
            mol_padding_mask:  [batch_size, max_mol_len]
            pocket_padding_mask:  [batch_size, max_pocket_len]
            cross_distance_predict:  [batch_size, max_mol_len, max_pocket_len]
        
        :
            ()
        """
        batch_size = mask.size(0)
        #print('batch_size:', batch_size)
        unmasked_matrices = []
        
        for i in range(batch_size):
            #print('i:', i)
            # 
            mol_mask = mask[i].bool()  # [max_mol_len]
            
            # 
            mol_len = mol_mask.sum().item()
            
            # 
            valid_mol_indices = mol_mask.nonzero().squeeze(-1)  # [mol_len]
            
            # 
            valid_matrix = matrix[i][valid_mol_indices]  # [mol_len, pocket_len]
            
            # 2
            valid_dist_matrix = valid_matrix[1:-1]
            #print('valid_dist_matrix:', valid_dist_matrix.shape)
            
            unmasked_matrices.append(valid_dist_matrix)
        #print('torch.stack(unmasked_matrices, dim = 0):', torch.stack(unmasked_matrices, dim = 0).shape)
        return torch.stack(unmasked_matrices, dim = 0)
    

    def forward_sample(self, model, sample, reduce=True):
        #，
    #try:
        ## ，0，,，，，
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"]) #holo_coord
        ##，256，？，，，,
        cross_distance_predict, holo_distance_predict, coord_predict, prmsd_predict, ligand_emb, pocket_emb, l_mask, p_mask, l_atom_num, p_atom_num = net_output[:10]
        #print('sample["pocket_name"]:', sample["pocket_name"])
        #print('cross_distance_predict, holo_distance_predict, coord_predict, prmsd_predict, ligand_emb, pocket_emb.shape:', cross_distance_predict.shape, holo_distance_predict.shape, coord_predict.shape, prmsd_predict.shape, ligand_emb.shape, pocket_emb.shape)
        
        
        #print('cross_distance_predict[0]:', cross_distance_predict[0])
        #print('holo_distance_predict[0]:', holo_distance_predict[0]) #nan
        #print('prmsd_predict[0]:', prmsd_predict[0])
        #print('pocket_emb[0]:', pocket_emb[0])
        
        tick = 0.25
        max_bins = 32 
        prmsd_logit = F.softmax(prmsd_predict.float(), dim=-1)   # BS, N, MAX_BINS
        weight = torch.arange(max_bins,).type_as(prmsd_logit).unsqueeze(0) + tick / 2
        prmsd_score = (prmsd_logit * weight).sum(dim=-1).mean(dim=-1)

        sample_size = sample["target"]["holo_coord"].size(0)
        logging_output = {
            "loss": 0,
            "cross_distance_loss": 0,
            "distance_loss": 0,
            "coord_loss": 0,
            "prmsd_loss": 0,
            "prmsd_score": 0,
            "bsz": sample_size,
            "sample_size": 1,
            "coord_predict": coord_predict.data,   # last iteration
            "coord_target": sample["target"]["holo_coord"].data,
        }

        ligand_shape = [sample_size, l_atom_num, 3]
        protein_shape = [sample_size, p_atom_num, 3]
        cross_distance_target_shape = [sample_size, l_atom_num, p_atom_num]
        
        if not self.training:
            logging_output["smi_name"]      = sample["smi_name"]
            logging_output["pocket_name"]   = sample["pocket_name"]
            #print('logging_output["smi_name"]:', logging_output["smi_name"])
            logging_output["coord_predict"] = coord_predict.data.detach().cpu()
            
            logging_output["prmsd_score"]   = prmsd_score.data.detach().cpu()
            #print('logging_output["prmsd_score"]:', logging_output["prmsd_score"].shape)
            
            logging_output["atoms"]         = self.get_unmasked_2d(l_mask, sample["net_input"]["mol_src_tokens"]).view(sample_size, l_atom_num).data.detach().cpu()
            logging_output["pocket_atoms"]  = self.get_unmasked_2d(p_mask, sample["net_input"]["pocket_src_tokens"]).view(sample_size, p_atom_num).data.detach().cpu()
            
            #print('logging_output["atoms"].shape:', logging_output["atoms"].shape)
            #print('logging_output["pocket_atoms"]:', logging_output["pocket_atoms"].shape)
            
            
            logging_output["coordinates"]   = self.get_unmasked_pos_emb_(l_mask, sample["net_input"]["mol_src_coord"]).view(ligand_shape).data.detach().cpu()

            #3, 0
            logging_output["holo_coordinates"]          = self.get_unmasked_pos_emb_(l_mask, sample["target"]["holo_coord"]).view(ligand_shape).data.detach().cpu()
            logging_output["holo_pocket_coordinates"]   = self.get_unmasked_pos_emb_(p_mask, sample["target"]["holo_coord_pocket"]).view(protein_shape).data.detach().cpu()
            logging_output["pocket_coordinates"]        = self.get_unmasked_pos_emb_(p_mask, sample["net_input"]["pocket_src_coord"]).view(protein_shape).data.detach().cpu()
            logging_output["cross_distance"]            = cross_distance_predict.data.detach().cpu() #，，     

            #print('sample["holo_center_coordinates"].data.detach().cpu():', sample["holo_center_coordinates"].data.detach().cpu().shape)

            logging_output["holo_center_coordinates"] = sample["holo_center_coordinates"].data.detach().cpu() #torch.Size([1, 8])
            
            #print('logging_output["holo_center_coordinates]":', logging_output["holo_center_coordinates"])
            #tensor([[ 4.7899, 31.4300, 10.1615,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
            #logging_output['ligand_emb'] = ligand_emb.data.detach().cpu() 
            #logging_output['pocket_emb'] = pocket_emb.data.detach().cpu()
    
            assert logging_output["cross_distance"].shape[1] == logging_output["coord_predict"].shape[1]
            assert logging_output["cross_distance"].shape[2] == logging_output["pocket_coordinates"].shape[1]
            assert logging_output["holo_coordinates"].shape[1] == logging_output["coord_predict"].shape[1]
            assert logging_output["holo_pocket_coordinates"].shape[1] == logging_output["pocket_coordinates"].shape[1]

        loss = 0
            
        #return loss, sample_size, (logging_output, sample['pocket_name'])
        return loss, sample_size, logging_output
    
    '''
    except Exception as e:
        #print('error:', e)
        ##print('error name:', sample['pocket_name'])
        #40rdkit，，，，，
        #raise Exception('error,stop')
        #return None, None, (None, sample['pocket_name'])
        return None, None, None
    '''
    
    
        

                
        
        



    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            f"{split}_loss", loss_sum / sample_size, sample_size, round=3
        )
        cross_distance_loss = sum(log.get("cross_distance_loss", 0) for log in logging_outputs)
        if cross_distance_loss > 0:
            metrics.log_scalar(
                "cross_distance_loss", cross_distance_loss / sample_size, sample_size, round=3
            )
        distance_loss = sum(log.get("distance_loss", 0) for log in logging_outputs)
        if distance_loss > 0:
            metrics.log_scalar(
                "distance_loss", distance_loss / sample_size, sample_size, round=3
            )
        coord_loss = sum(log.get("coord_loss", 0) for log in logging_outputs)
        if coord_loss > 0:
            coord_predict = [log.get("coord_predict")[i].cpu().numpy() for log in logging_outputs for i in range(log.get("coord_predict").size(0))]
            coord_target = [log.get("coord_target")[i].cpu().numpy() for log in logging_outputs for i in range(log.get("coord_target").size(0))]
            metrics.log_scalar(
                "coord_loss", coord_loss / sample_size, sample_size, round=3
            )
            rmsd_list = [RMSD(_predict, _target) for _predict,_target in zip(coord_predict, coord_target)]
            metrics.log_scalar(
                "RMSD", np.mean(rmsd_list), sample_size, round=3
            )
        prmsd_loss = sum(log.get("prmsd_loss", 0) for log in logging_outputs)
        if prmsd_loss > 0:
            metrics.log_scalar(
                "prmsd_loss", prmsd_loss / sample_size, sample_size, round=3
            )
            
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


def RMSD(coord_predict, coord_target):
    mask = coord_target != 0 #
    rmsd = np.sqrt(np.sum(((coord_predict - coord_target) ** 2) * mask) / (mask[:,0].sum()))
    return rmsd
