import torch
from .MMFF import MMFF_Energy, combine_coords_with_masks, combine_MMFF_params_from_mols,combine_fix_masks_with_masks, MMFF_keys, MMFF_pad_dim
from ..comparm import FGP
from comparm import GP
import torch.nn.functional as F

class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev)

class CoorMin(torch.nn.Module):
    def __init__(self):
        super(CoorMin, self).__init__()
        self.MMFF_lossfunction = MMFF_Energy(split_interact=False, warm=False)
        self.lr = FGP.MMFF_lr
        self.decay = FGP.MMFF_decay
        self.max_decay_step = FGP.MMFF_max_decay_step
        self.patience_tol_step = FGP.MMFF_patience_tol_step
        self.patience_tol_value = FGP.MMFF_patience_tol_value
        self.clip = FGP.MMFF_clip
        self.constraint = FGP.MMFF_constraint

    def forward(self, l_coor_pred, mmff_params, fix_masks, loop=None, constraint=None, show_state=False, min_type='SGD',return_delta=False, cross_loss = None, cross_distance = None, ligand_masks = None, protein_masks = None):
        with TemporaryGrad():  # i.e. torch.set_grad_enabled(True)
            coor_min,energy_min = self.FF_min(l_coor_pred, mmff_params,fix_masks=fix_masks,
            loop=self.loop if isinstance(loop, type(None)) else loop,
            lr=self.lr,
            decay=self.decay, max_decay_step=self.max_decay_step,
            patience_tol_step=self.patience_tol_step, patience_tol_value=self.patience_tol_value,
            clip=self.clip,
            constraint=self.constraint if isinstance(constraint, type(None)) else constraint,
            show_state=show_state, min_type=min_type, cross_loss = cross_loss, 
            cross_distance = cross_distance, ligand_masks = ligand_masks, protein_masks = protein_masks,
            )
        if torch.isnan(coor_min).any():
            if return_delta:
                return torch.zeros_like(l_coor_pred).to(l_coor_pred.device),energy_min
            else:
                return l_coor_pred,energy_min
        else:
            if return_delta:
                return coor_min - l_coor_pred, energy_min 
            else:
                return coor_min,energy_min 

    def FF_min(self, coor_pred, mmff_params,fix_masks=None,
            loop=10000, lr=5e-5,
            decay=0.5, max_decay_step=10,
            patience_tol_step=10, patience_tol_value=0,
            clip=1e+5, constraint=0, show_state=False, min_type='SGD', cross_loss = None,
            cross_distance = None, ligand_masks = None, protein_masks = None):
        
        #
        #ligand_pos.requires_grad = True
        #ligand_pos_Parm = torch.nn.Parameter(ligand_pos.detach().clone()).to(ligand_pos.device)


        #
        coor_pred_detach = coor_pred.detach()
        coor_paramed = torch.nn.Parameter(coor_pred.detach().clone()).to(coor_pred.device)

        #ligand_pos_Parm, protein_pos = coor_paramed[ligand_masks], coor_paramed[protein_masks]

        if min_type == 'SGD':
            optimizer = torch.optim.SGD([coor_paramed], lr=0.1)
        elif min_type =='AdamW':
            optimizer = torch.optim.AdamW([coor_paramed], lr=0.1)
        elif min_type == 'LBFGS':
            optimizer = torch.optim.LBFGS([coor_paramed], lr=1.0, line_search_fn='strong_wolfe') #lr = 0.0001
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay)



        #print('loop:', loop) #1
        #exit()
        for step in range(loop):
            def closure():
                ligand_pos_Parm, protein_pos = coor_paramed[ligand_masks], coor_paramed[protein_masks]

                #pred_distance = torch.cdist(ligand_pos_Parm,protein_pos,compute_mode='donot_use_mm_for_euclid_dist') #use_mm_for_euclid_dist
                #print('pred_distance, cross_distance:', pred_distance.shape, cross_distance.shape) 
                #pred_distance, cross_distance: torch.Size([130, 1250]) torch.Size([13, 125])
                #red_distance, cross_distance: torch.Size([13, 125]) torch.Size([13, 125]) 

                if GP.cross_loss != 0:
                    # 
                    ligand_atom_num  = cross_distance[0].shape[0]
                    protein_atom_num = cross_distance[0].shape[1]

                    shape1 = ligand_pos_Parm.view(-1, ligand_atom_num, 3).shape
                    shape2 = protein_pos.view(-1, protein_atom_num, 3).shape
                    #print(shape1)
                    #print(shape2)
                    pred_distance = torch.empty([shape1[0], shape1[1], shape2[1]])  # 

                    new_cross_distance = torch.empty([shape1[0], shape1[1], shape2[1]])  # 


                    for i in range(shape1[0]):
                        pred_distance[i] = torch.cdist(ligand_pos_Parm.view(shape1)[i], protein_pos.view(shape2)[i], compute_mode='donot_use_mm_for_euclid_dist')
                        new_cross_distance[i] = cross_distance[i].clone().detach()
                    
                    if not pred_distance.requires_grad:
                        raise Exception('Grad error')

                    cross_loss = F.mse_loss(pred_distance, new_cross_distance)
                    
                else:
                    cross_loss = torch.tensor(0.0).cuda()
                    
                #print('cross_loss.requires_grad:', cross_loss.requires_grad) # true
                #print('cross_loss:', f'{cross_loss.detach().cpu().numpy():.5f}') #3.9178
                #print('pred_distance, new_cross_distance:', pred_distance.shape, new_cross_distance.shape) 
                #exit()

                if GP.force_loss == 0:
                    e = torch.tensor(0.0).cuda()
                else:
                    e = self.MMFF_lossfunction(coor_paramed, mmff_params, return_sum=True)
                    #print('e.shape:', e.shape) #orch.Size([batch, 1])
                    #print('cross_loss:', cross_loss.item)
                    #exit()
                    e = e.mean()  #e = e.sum() / mean()   #。l = e + d， 


                all_loss = e * GP.force_loss + cross_loss * GP.cross_loss #，，10, ，3.942e+03，，

                print(f"e_loss: {e.detach().cpu().numpy():.5f}, cross_loss: {(cross_loss * GP.cross_loss).detach().cpu().numpy():.5f}")

                if constraint > 0:
                    e_constraint = 0.5 * ((coor_paramed - coor_pred_detach) ** 2 * constraint).sum()
                    all_loss = all_loss + e_constraint

                optimizer.zero_grad()
                all_loss.backward(retain_graph=True)
                if min_type == 'SGD' or min_type =='AdamW':
                    torch.nn.utils.clip_grad_norm_([coor_paramed], clip)
                return all_loss
            
            e = closure()

            if min_type == 'SGD' or min_type =='AdamW':
                optimizer.step()
            elif min_type == 'LBFGS':
                optimizer.step(closure) #()，，
            #coord_paramed = coor_pred_detach*fix_masks.long()+coor_paramed*(1-fix_masks.long())
            if step == 0:
                e_min = e
                patience_sum = 0
                decay_step = 0
            elif e_min - e < patience_tol_value:
                patience_sum += 1
            else:
                e_min = e
            if patience_sum >= patience_tol_step:
                if decay_step < max_decay_step:
                    scheduler.step()
                    patience_sum = 0
                decay_step += 1
            if torch.isnan(e).any():
                break

            if show_state:
                print(f"Step:{step + 1}, lr:{optimizer.param_groups[0]['lr']:.5f}, ALL:{e.detach().cpu().numpy():.5f}")
                #exit()
                #print (e_split)
        
        
        coor_pred_min = coor_paramed.detach().clone().to(coor_pred.device)
        if GP.force_loss == 0:
            energy_min = 0.0
        else:
            energy_min=self.MMFF_lossfunction(coor_pred_min, mmff_params, return_sum=True)
            
        del optimizer
        del scheduler
        del coor_paramed
        torch.cuda.empty_cache()
        return coor_pred_min,energy_min
    
def opt_coords_moves(coords_pred,mols,coords_masks,fix_masks=None,loop=1,show_state=False,min_type='LBFGS',):
    coords_shape=coords_pred.shape
    coords_masks_=coords_masks.view(-1)
    coords_selected=combine_coords_with_masks(coords_pred,coords_masks)
    coords_fix_masks=combine_fix_masks_with_masks(fix_masks,coords_masks)

    mmff_params=combine_MMFF_params_from_mols(mols)

    for key in mmff_params.keys():
        mmff_params[key]=mmff_params[key].to(coords_pred.device)
    coord_opt=CoorMin().to(coords_pred.device)
    coord_moves,energy_min=coord_opt(coords_selected,mmff_params,fix_masks=coords_fix_masks,loop=loop,show_state=show_state,min_type=min_type,return_delta=True,)
    coords_moves_=torch.zeros(coords_shape).to(coords_pred.device)
    coords_moves_=coords_moves_.view(-1,3)
    coords_moves_[coords_masks_]=coord_moves
    coords_moves_=coords_moves_.view(*coords_shape)
    return coords_moves_,energy_min

def combine_pocket_masks_with_masks(pocket_masks,coords_masks):
    pocket_masks_=pocket_masks.view(-1)
    coords_masks_=coords_masks.view(-1)
    coords_pocket_masks=pocket_masks_[coords_masks_]
    return coords_pocket_masks

def combine_ligand_masks_with_masks(ligand_masks,coords_masks):
    ligand_masks_=ligand_masks.view(-1)
    coords_masks_=coords_masks.view(-1)
    coords_ligand_masks=ligand_masks_[coords_masks_]
    return coords_ligand_masks

def opt_complex_coords_moves_old(coords_pred,mols,coords_masks,fix_masks=None,loop=1,show_state=False,min_type='LBFGS',pocket_masks=None,ligand_masks=None):

    #，32，；
    #，2，3
    coords_shape=coords_pred.shape
    coords_masks_=coords_masks.view(-1)
    coords_selected=combine_coords_with_masks(coords_pred,coords_masks)

    coords_fix_masks=combine_fix_masks_with_masks(fix_masks,coords_masks)

    coords_pocket_masks=combine_pocket_masks_with_masks(pocket_masks,coords_masks)
    
    coords_ligand_masks=combine_ligand_masks_with_masks(ligand_masks,coords_masks)

    mmff_params=combine_MMFF_params_from_mols(mols)
    for key in mmff_params.keys():
        mmff_params[key]=mmff_params[key].to(coords_pred.device)

    coord_opt=CoorMin().to(coords_pred.device)

    #，：x = x + coord_moves，coord_moves（），
    #x = x + coord_moves，+
    coord_moves,energy_min=coord_opt(coords_selected,mmff_params,fix_masks=coords_fix_masks,loop=loop,show_state=show_state,min_type=min_type,return_delta=True,)

    #
    coords_moves_=torch.zeros(coords_shape).to(coords_pred.device)
    coords_moves_=coords_moves_.view(-1,3)
    coords_moves_[pocket_masks.view(-1)]=coord_moves[coords_pocket_masks]
    coords_moves_[ligand_masks.view(-1)]=coord_moves[coords_ligand_masks]
    #coords_moves_[coords_masks_]=coord_moves
    coords_moves_=coords_moves_.view(*coords_shape)
    return coords_moves_,energy_min



def opt_complex_coords_moves(coords_pred,mols,coords_masks,loop=1,show_state=False,min_type='LBFGS',mask_ligand=None,cross_loss=None, ligand_pos = None, protein_pos = None, cross_distance = None):

    #，32，；
    #，2，3

    '''
    coords_shape=coords_pred.shape
    coords_masks_=coords_masks.view(-1)
    coords_selected=combine_coords_with_masks(coords_pred,coords_masks)

    coords_fix_masks=combine_fix_masks_with_masks(fix_masks,coords_masks)

    coords_pocket_masks=combine_pocket_masks_with_masks(pocket_masks,coords_masks)
    
    coords_ligand_masks=combine_ligand_masks_with_masks(ligand_masks,coords_masks)
    '''
    assert coords_masks.shape == mask_ligand.shape
    ligand_masks  = mask_ligand == 1
    protein_masks = mask_ligand == 0
    coords_fix_masks = protein_masks
    coords_selected  = coords_pred



    mmff_params=combine_MMFF_params_from_mols(mols)
    for key in mmff_params.keys():
        mmff_params[key]=mmff_params[key].to(coords_pred.device)

    coord_opt=CoorMin().to(coords_pred.device)

    #，：x = x + coord_moves，coord_moves（），
    #x = x + coord_moves，+
    coord_moves,energy_min=coord_opt(coords_selected,mmff_params,fix_masks=coords_fix_masks,loop=loop,show_state=show_state,min_type=min_type,return_delta=True, cross_loss = cross_loss, cross_distance = cross_distance, ligand_masks = ligand_masks, protein_masks = protein_masks)
    
    #print('coords_selected:', coords_selected.shape)
    #print('coord_moves:', coord_moves.shape)
    #print('ligand_masks:', ligand_masks.shape)
    coord_moves = coord_moves[ligand_masks]

    '''
    #
    coords_moves_=torch.zeros(coords_shape).to(coords_pred.device)
    coords_moves_=coords_moves_.view(-1,3)
    coords_moves_[pocket_masks.view(-1)]=coord_moves[coords_pocket_masks]
    coords_moves_[ligand_masks.view(-1)]=coord_moves[coords_ligand_masks]
    #coords_moves_[coords_masks_]=coord_moves
    coords_moves_=coords_moves_.view(*coords_shape)
    '''

    return coord_moves, energy_min