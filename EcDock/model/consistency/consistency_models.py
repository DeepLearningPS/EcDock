import math
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm



from typing import Iterator

from torch import Tensor, nn
from EcDock.comparm import *

import random
import numpy as np
from torch_scatter import scatter_sum, scatter_mean
import torch.nn.functional as F
import copy
import random

import networkx as nx
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit import Geometry
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
#from models.molopt_score_model import center_pos, index_to_log_onehot, q_v_sample

from rdkit.Geometry.rdGeometry import Point3D
import time

from EcDock.utils.utils_torch import *

from TreeInvent2.model.consistency.consistency_models import opt_coords_moves,opt_complex_coords_moves


np.random.seed(2023)
torch.manual_seed(2023)
random.seed(2023)
torch.cuda.manual_seed_all(2023)



def Change_Mol_D3coord(inputmol,coords):
    '''
        ，inputmol，coords：
    '''
    molobj=copy.deepcopy(inputmol)
    conformer=molobj.GetConformer()
    id=conformer.GetId()
    for cid,xyz in enumerate(coords):
        ##print(xyz[0],xyz[1],xyz[2],type(xyz))
        conformer.SetAtomPosition(cid,Point3D(float(xyz[0]),float(xyz[1]),float(xyz[2]))) #
    conf_id=molobj.AddConformer(conformer)
    molobj.RemoveConformer(id)
    molobj = Chem.RemoveHs(molobj)
    return molobj



def pad_dims_like(x: Tensor, other: Tensor) -> Tensor:
    """Pad dimensions of tensor `x` to match the shape of tensor `other`.

    Parameters
    ----------
    x : Tensor
        Tensor to be padded.
    other : Tensor
        Tensor whose shape will be used as reference for padding.

    Returns
    -------
    Tensor
        Padded tensor with the same shape as other.
    """
    ndim = other.ndim - x.ndim
    return x.view(*x.shape, *((1,) * ndim))


def _update_ema_weights_old(
    ema_weight_iter: Iterator[Tensor],
    online_weight_iter: Iterator[Tensor],
    ema_decay_rate: float,
) -> None:
    for ema_weight, online_weight in zip(ema_weight_iter, online_weight_iter):
        if ema_weight.data is None:
            ema_weight.data.copy_(online_weight.data)
        else:
            ema_weight.data.lerp_(online_weight.data, 1.0 - ema_decay_rate)





def _update_ema_weights(
    ema_weight_iter: Iterator[Tensor],
    online_weight_iter: Iterator[Tensor],
    ema_decay_rate: float,
    mode = None
) -> None:
    for ema_weight, online_weight in zip(ema_weight_iter, online_weight_iter):
        if ema_weight.data is None:
            ema_weight.data.copy_(online_weight.data)
        else:
            try:
                ema_weight.data.lerp_(online_weight.data, 1.0 - ema_decay_rate) #，，，
            except Exception as e:
                if mode == 'buffers':
                    #print'error:', e)
                    #print'ema_decay_rate:', ema_decay_rate)
                    #print'ema_weight.data:', ema_weight.data)
                    #print'online_weight.data:', online_weight.data)
                    #print'skip update buffers')
                    #，
                    #if online_model != None:
                        ##print'online_model:', online_model)
                    #exit()
                    pass
                else:
                    #print'error:', e)
                    #print'ema_decay_rate:', ema_decay_rate)
                    #print'ema_weight.data:', ema_weight.data)
                    #print'online_weight.data:', online_weight.data)
                    #print'error update params')
                    #，
                    #if online_model != None:
                        ##print'online_model:', online_model)
                    exit()


def update_ema_model(
    ema_model: nn.Module, online_model: nn.Module, ema_decay_rate: float
) -> nn.Module:
    """Updates weights of a moving average model with an online/source model.

    Parameters
    ----------
    ema_model : nn.Module
        Moving average model.
    online_model : nn.Module
        Online or source model.
    ema_decay_rate : float
        Parameter that controls by how much the moving average weights are changed.

    Returns
    -------
    nn.Module
        Updated moving average model.
    """
    # Update parameters
    _update_ema_weights(
        ema_model.parameters(), online_model.parameters(), ema_decay_rate
    )
    #exit()
    ##print'online_model:', online_model)
    # Update buffers
    #long，？？？
    _update_ema_weights(ema_model.buffers(), online_model.buffers(), ema_decay_rate, mode = 'buffers')

    return ema_model
    
def timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 25,
) -> int:
    """Implements the proposed timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.
    """
    num_timesteps = final_timesteps**2 - initial_timesteps**2
    num_timesteps = current_training_step * num_timesteps / total_training_steps
    num_timesteps = math.ceil(math.sqrt(num_timesteps + initial_timesteps**2) - 1)

    return num_timesteps + 1  #[initial_timesteps,final_timesteps],final_timesteps25


def ema_decay_rate_schedule(
    num_timesteps: int, initial_ema_decay_rate: float = 0.95, initial_timesteps: int = 2
) -> float:
    """Implements the proposed EMA decay rate schedule.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    initial_ema_decay_rate : float, default=0.95
        EMA rate at the start of training.
    initial_timesteps : int, default=2
        Timesteps at the start of training.

    Returns
    -------
    float
        EMA decay rate at the current point in training.
    """
    return math.exp(
        (initial_timesteps * math.log(initial_ema_decay_rate)) / num_timesteps
    )


def karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = None,
) -> Tensor:
    """Implements the karras schedule that controls the standard deviation of
    noise added.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    sigma_min : float, default=0.002
        Minimum standard deviation.
    sigma_max : float, default=80.0
        Maximum standard deviation
    rho : float, default=7.0
        Schedule hyper-parameter.
    device : torch.device, default=None
        Device to generate the schedule/sigmas/boundaries/ts on.

    Returns
    -------
    Tensor
        Generated schedule/sigmas/boundaries/ts.
    """
    rho_inv = 1.0 / rho
    # Clamp steps to 1 so that we don't get nans
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min**rho_inv + steps * (
        sigma_max**rho_inv - sigma_min**rho_inv
    )
    sigmas = sigmas**rho

    return sigmas


def skip_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the residual connection.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the residual connection.
    """
    return sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)


def output_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the model's output.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the model's output.
    """
    return (sigma_data * (sigma - sigma_min)) / (sigma_data**2 + sigma**2) ** 0.5


def model_forward_wrapper(
    model: nn.Module,
    feats: Tensor = None,
    adjs: Tensor = None,
    xyzs: Tensor = None,
    gmasks: Tensor = None,
    sigma: Tensor = None,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,

    protein_pos=None, #
    protein_v=None, 
    batch_protein=None,

    init_ligand_pos=None, #
    init_ligand_v=None,  #
    batch_ligand=None,
    time_step=None,
    
    org_ligand_pos = None,
    org_protein_pos = None,
    ligand_bond_index = None, ligand_bond_type = None, ligand_bond_type_batch = None,
    protein_max_atom_num = None, ligand_max_atom_num  = None,
    protein_element = None, ligand_element = None,

    ligand_atom_isring  =  None,
    ligand_atom_isO     =  None,
    ligand_atom_isN     =  None,

    protein_atom_isring =  None,
    protein_atom_isO    =  None,
    protein_atom_isN    =  None,


    cross_lig_isring_flag   = None,
    cross_lig_isO_flag      = None,
    cross_lig_isN_flag      = None,

    cross_pro_isring_flag   = None,
    cross_pro_isO_flag      = None,
    cross_pro_isN_flag      = None,

    cross_ligand    = None,
    cross_protein   = None,
    cross_distance  = None,

    cross_bond_index = None, 
    cross_bond_type = None, 
    cross_bond_index_reverse = None, 
    cross_bond_type_reverse = None,

    protein_coords_predict = None,

    complex_mol = None,

    protein_element_batch = None,
    protein_link_t_batch = None,
    protein_link_t_reverse_batch = None,

    ligand_element_batch = None,

    rd_pos = None,

    sample=False,
    scale = True,
    rate = 10.0,
    scale_step = None, 

) -> Tensor:
    """Wrapper for the model call to ensure that the residual connection and scaling
    for the residual and output values are applied.

    Parameters
    ----------
    model : nn.Module
        Model to call.
    x : Tensor
        Input to the model, e.g: the noisy samples.
    sigma : Tensor
        Standard deviation of the noise. Normally referred to as t.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    **kwargs : Any
        Extra arguments to be passed during the model call.

    Returns
    -------
    Tensor
        Scaled output from the model with the residual connection applied.
    """
    c_skip = skip_scaling(sigma, sigma_data, sigma_min)
    c_out = output_scaling(sigma, sigma_data, sigma_min)

    # Pad dimensions as broadcasting will not work
    c_skip = c_skip.index_select(0, batch_ligand).view([-1, 1]).to(xyzs.device)
    c_out  = c_out.index_select(0, batch_ligand).view([-1, 1]).to(xyzs.device)
    #print ('*',c_out.shape,c_skip.shape)
    sigma=sigma.to(xyzs.device)


    st = time.perf_counter()
    preds = model(
        protein_pos=protein_pos, 
        protein_v=protein_v, 
        batch_protein=batch_protein,

        init_ligand_pos=init_ligand_pos, #
        init_ligand_v=init_ligand_v,  #
        batch_ligand=batch_ligand,
        time_step=time_step,
        sample = sample,
        org_ligand_pos = org_ligand_pos,
        org_protein_pos = org_protein_pos,
        ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
        sigmas = sigma,
        protein_element = protein_element, ligand_element = ligand_element,

        ligand_atom_isring  = ligand_atom_isring,
        ligand_atom_isO     = ligand_atom_isO,
        ligand_atom_isN     = ligand_atom_isN,

        protein_atom_isring = protein_atom_isring,
        protein_atom_isO    = protein_atom_isO,
        protein_atom_isN    = protein_atom_isN,

        cross_lig_isring_flag   = cross_lig_isring_flag,
        cross_lig_isO_flag      = cross_lig_isO_flag,
        cross_lig_isN_flag      = cross_lig_isN_flag,

        cross_pro_isring_flag   = cross_pro_isring_flag,
        cross_pro_isO_flag      = cross_pro_isO_flag,
        cross_pro_isN_flag      = cross_pro_isN_flag,

        cross_ligand    = cross_ligand,
        cross_protein   = cross_protein,
        cross_distance  = cross_distance,


        cross_bond_index = cross_bond_index, 
        cross_bond_type = cross_bond_type, 
        cross_bond_index_reverse = cross_bond_index_reverse, 
        cross_bond_type_reverse = cross_bond_type_reverse,

        protein_coords_predict = protein_coords_predict,

        complex_mol = complex_mol,

        protein_element_batch = protein_element_batch,
        protein_link_t_batch = protein_link_t_batch,
        protein_link_t_reverse_batch = protein_link_t_reverse_batch,
    
        ligand_element_batch = ligand_element_batch,

        rd_pos = rd_pos,


    )

    end = time.perf_counter()
    #print('a model time s:', round(end - st, 4)) #0.2，3

    #if sample == True:
        #print('cskip:', c_skip[0])
        #print('cout:',c_out[0])
        #print('---------------------------------------------')

    preds['pred_ligand_pos'] = c_skip * xyzs + c_out * preds['pred_ligand_pos'] 
    preds['final_pos'][preds['mask_ligand']] = preds['pred_ligand_pos'] #



    return preds



def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0) #，？
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    elif mode == 'ligand':
        offset = scatter_mean(ligand_pos, batch_ligand, dim=0) #，？
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset





def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)  #，ont-hot，num——classes
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))   #，，01e-30，
    return log_x



def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return gumbel_noise + logits


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)



def to_torch_const(x):
    ##print'x:', x)
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x




def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)



def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):  #kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ] ，z0，zt
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand, protein_element, ligand_element):
    # previous version has problems when ligand atom types are fixed
    # (due to sorting randomly in case of same element)

    #，
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0) #，
    # sort_idx = batch_ctx.argsort()
    sort_idx = torch.sort(batch_ctx, stable=True).indices  #，sort_idx
    #，

    #，，，
    mask_ligand = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]   #0，1

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    element_ctx = torch.cat([protein_element, ligand_element], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx, element_ctx, mask_ligand



def add_random_offset(mol):
    for atom in mol.GetAtoms():
        pos = atom.GetIdx()
        for _ in range(3):
            atom_pos = mol.GetConformer().GetAtomPosition(pos)
            mol.GetConformer().SetAtomPosition(pos, (atom_pos.x + random.uniform(-0.01, 0.01), 
                                                    atom_pos.y + random.uniform(-0.01, 0.01), 
                                                    atom_pos.z + random.uniform(-0.01, 0.01)))


def GetDihedral(conf, atom_idx): #
    #print('conf:', conf)
    #print('atom_idx:', atom_idx)
    #print('type(atom_idx[0]):', type(atom_idx[0]))
    #rdMolTransforms.GetDihedralRadmolmol
    #print('conf num:', len(conf.GetConformers()))#conf.GetConformers()[0]conf.GetConformer()

    torsion_degrees = rdMolTransforms.GetDihedralRad(conf.GetConformer(), atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])#[0,2pi]
    #rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3]) #，[0,360]
        
    t = torsion_degrees // (2 * np.pi) #，
    torsion_degrees = torsion_degrees - (2 * np.pi) * t
    #print(f'Torsion {torsion.GetIdx()}: {torsion_degrees:.2f} degrees')

    return torsion_degrees

def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def get_torsion_angles(mol): #list
    torsions_list = []
    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)
    nodes = set(G.nodes())
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        l = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append(
            (n0[0], e[0], e[1], n1[0])
        )
    return torsions_list


def compute_tor(mol):
    mol_ = copy.deepcopy(mol)
    mol_maybe_noh = RemoveHs(mol_, sanitize=True)
    rotable_bonds = get_torsion_angles(mol_maybe_noh) #4
    tor_list = []
    for i in rotable_bonds:
        tor = GetDihedral(mol_maybe_noh, i)
        tor_list.append(tor)

    tor = tor_list
    return tor



def internal_coordinate(mol_list):
    # 
    bond_lists       = []
    angle_lists      = []
    torsion_lists    = []

    for mol in mol_list: #mol
        bond_list       = []
        angle_list      = []
        torsion_list    = []

        for bond in mol.GetBonds():
            bond_length = Chem.rdMolTransforms.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            #print(f'Bond {bond.GetIdx()}: {bond_length:.2f} angstroms')
            bond_list.append(bond_length)



        # 
        for atom in mol.GetAtoms():
            atom_id = atom.GetIdx()
            atom_neighbors = atom.GetNeighbors()
            if len(atom_neighbors) == 2:
                angle_value = Chem.rdMolTransforms.GetAngleRad(mol.GetConformer(), atom_neighbors[0].GetIdx(), atom_id, atom_neighbors[1].GetIdx())
                angle_degrees = angle_value 
                #
                t = angle_degrees // (2 * np.pi) #，
                angle_degrees = angle_degrees - (2 * np.pi) * t
                #print(f'Angle {angle.GetIdx()}: {angle_degrees:.2f} degrees')
                angle_list.append(angle_degrees)


        #add_random_offset(mol)
        # 
        torsion_list = compute_tor(mol)

        
        bond_lists.extend(bond_list)
        angle_lists.extend(angle_list)
        torsion_lists.extend(torsion_list)
    
    return torch.stack(bond_lists), torch.stack(angle_lists), torch.stack(torsion_lists)



def calc_performance_stats(true_mols, model_mols):

    rmsd_list = []
    for tc, mc in zip(true_mols, model_mols):
        try:
            rmsd_val = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
        except RuntimeError:
            print('rmsd error, skip')
            pass
        rmsd_list.append(rmsd_val)


    rmsd = np.array(rmsd_list).mean()

    return rmsd


def distance(coords):
    #，，

    A = coords.unsqueeze(0)  #  batch 
    B = coords.unsqueeze(1)  # 
    # 
    dist_vectors = A - B
    # 
    dist_sq = torch.norm(dist_vectors, dim=-1)

    #print('dist_sq:', dist_sq.shape) #n * n，，
    return dist_sq.view(-1)


def py_rmsd(coords1, coords2):
    # 
    squared_diff = torch.sum((coords1 - coords2) ** 2, dim=1)
    
    # 
    rmsd = torch.sqrt(torch.mean(squared_diff))
    
    return rmsd
    

def compute_rmsd(coords1, coords2):
    # 
    center1 = torch.mean(coords1, dim=0)
    center2 = torch.mean(coords2, dim=0)
    
    # 
    coords1_centered = coords1 - center1
    coords2_centered = coords2 - center2
    
    # 
    U, _, Vt = torch.svd(torch.matmul(coords1_centered.t(), coords2_centered))
    rotation_matrix = torch.matmul(U, Vt)
    
    # 
    coords1_rotated = torch.matmul(coords1_centered, rotation_matrix)
    
    # 
    squared_diff = torch.sum((coords1_rotated - coords2_centered) ** 2, dim=1)
    
    # 
    rmsd = torch.sqrt(torch.mean(squared_diff))
    
    return rmsd





def improved_timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 25,
) -> int:
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    total_training_steps_prime = math.floor(
        total_training_steps
        / (math.log2(math.floor(final_timesteps / initial_timesteps)) + 1)
    )
    num_timesteps = initial_timesteps * math.pow(
        2, math.floor(current_training_step / total_training_steps_prime)
    )
    num_timesteps = min(num_timesteps, final_timesteps) + 1

    return num_timesteps


def improved_loss_weighting(sigmas: Tensor) -> Tensor:
    """Computes the weighting for the consistency loss.

    Parameters
    ----------
    sigmas : Tensor
        Standard deviations of the noise.

    Returns
    -------
    Tensor
        Weighting for the consistency loss.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    return 1 / (sigmas[1:] - sigmas[:-1])



def pseudo_huber_loss(input: Tensor, target: Tensor, batch_ligand) -> Tensor:
    #huber
    """Computes the pseudo huber loss.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    target : Tensor
        Target tensor.

    Returns
    -------
    Tensor
        Pseudo huber loss.
    """
    c = 0.00054 * math.sqrt(math.prod(input.shape[1:])) #list, ，，3

    loss = scatter_mean((torch.sqrt((input - target) ** 2 + c**2) - c).sum(-1), batch_ligand, dim=0) #

    return loss





def lognormal_timestep_distribution(
    num_samples: int,
    sigmas: Tensor,
    mean: float = -1.1,
    std: float = 2.0,
) -> Tensor:
    """Draws timesteps from a lognormal distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    sigmas : Tensor
        Standard deviations of the noise.
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.

    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()

    timesteps = torch.multinomial(pdf, num_samples, replacement=True)

    return timesteps




class ConsistencyTraining:
    """Implements the Consistency Training algorithm proposed in the paper.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_max : float, default=80.0
        Maximum standard deviation of the noise.
    rho : float, default=7.0
        Schedule hyper-parameter.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    initial_timesteps : int, default=2
        Schedule timesteps at the start of training.
    final_timesteps : int, default=150
        Schedule timesteps at the end of training.
    initial_ema_decay_rate : float, default=0.95
        EMA rate at the start of training.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        initial_timesteps: int = 2, #2
        final_timesteps: int = 25, ##self.final_timesteps，150，，25/15，，，

        lognormal_mean: float = -1.1,
        lognormal_std: float = 2.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps

        self.lognormal_mean = lognormal_mean
        self.lognormal_std = lognormal_std



    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_prob = log_sample_categorical(log_qvt_v0)
        sample_index = sample_prob.argmax(dim=-1) #，
        log_sample = index_to_log_onehot(sample_index, self.num_classes) #one-hot，
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, v0) * q(vt-1 | v0) / q(vt | v0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0
    


    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs


    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0 #
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v
    

    def calculate_distance_matrix(self, A, B):
        """
        

        :
        A (torch.Tensor):  (n, 3) 
        B (torch.Tensor):  (m, 3) 

        :
        torch.Tensor:  (n, m) 
        """
        # A  (n, 3)，B  (m, 3)
        #  A  B 
        diff = A.unsqueeze(1) - B.unsqueeze(0)  # diff  (n, m, 3)
        dist_matrix = torch.sqrt(torch.sum(diff**2, dim=2))  # dist_matrix  (n, m)
        return dist_matrix



    def calculate_distance_matrix_batch(self, n, m):

        #  n  m  [batch_size, n, 3]  [batch_size, m, 3] 
        #batch_size = 4
        #n = 5
        #m = 6

        # 
        #n = torch.randn(batch_size, n, 3)  #  n,  [batch_size, n, 3]
        #m = torch.randn(batch_size, m, 3)  #  m,  [batch_size, m, 3]

        # 
        # 1. 
        n_expanded = n.unsqueeze(2)  #  [batch_size, n, 1, 3]
        m_expanded = m.unsqueeze(1)  #  [batch_size, 1, m, 3]

        # 2. 
        distance_matrix = torch.sqrt(torch.sum((n_expanded - m_expanded) ** 2, dim=-1))

        # ， [batch_size, n, m]， `n`  `m` 
        #print(distance_matrix.shape)  # [batch_size, n, m]
        #print(distance_matrix)
        return distance_matrix



    def IC_Loss(self,pred,target,zmats,gmasks):
        #print('pred,target,zmats,gmasks:', pred.shape,target.shape,zmats.shape,gmasks.shape)
        #pred,target,zmats,gmasks: torch.Size([2, 250, 3]) torch.Size([2, 250, 3]) torch.Size([2, 250, 5]) torch.Size([2, 250])
        #exit()
        #pred,target,zmats,gmasks: torch.Size([900, 9, 3]) torch.Size([900, 9, 3]) torch.Size([900, 9, 5]) torch.Size([900, 9])
        pred_bonddis,pred_angle,pred_dihedral,j1,j2,j3=xyz2ic(pred,zmats) #
        target_bonddis,target_angle,target_dihedral,j1,j2,j3=xyz2ic(target,zmats)
        pred_dismat=torch.cdist(pred,pred,compute_mode='donot_use_mm_for_euclid_dist')
        target_dismat=torch.cdist(target,target,compute_mode='donot_use_mm_for_euclid_dist')
        gmasks_2D=gmasks.unsqueeze(-1)*gmasks.unsqueeze(-1).permute(0,2,1)
        loss_angle=F.mse_loss(pred_angle[gmasks],target_angle[gmasks])
        loss_dismat=F.mse_loss(pred_dismat[gmasks_2D],target_dismat[gmasks_2D])
        loss_bonddis=F.mse_loss(pred_bonddis[gmasks],target_bonddis[gmasks])
        dihedral_diff=torch.abs(pred_dihedral[gmasks]-target_dihedral[gmasks])
        dihedral_diff=torch.where(dihedral_diff>math.pi,math.pi*2-dihedral_diff,dihedral_diff)
        loss_dihedral=torch.mean(torch.square(dihedral_diff))
        return loss_dismat,loss_bonddis,loss_angle,loss_dihedral



    def __call__(
        self,
        online_model: nn.Module,
        ema_model: nn.Module,
        current_training_step: int,
        total_training_steps: int, 
        args,
        config,
        protein_atom_feature_dim,
        ligand_atom_feature_dim,

        protein_pos=None,
        protein_v=None,
        affinity=None,
        batch_protein=None,
        
        ligand_pos=None,
        ligand_v=None,
        batch_ligand=None,

        ligand_bond_index=None,
        ligand_bond_type=None,
        ligand_bond_type_batch=None,

        protein_element=None, 
        ligand_element=None,

        ligand_mol=None,

        ligand_fill_coords =  None,
        ligand_fill_zmats  =  None,
        ligand_fill_masks  =  None,
        ligand_fill_atom_order = None,

        ligand_atom_isring  =  None,
        ligand_atom_isO     =  None,
        ligand_atom_isN     =  None,

        protein_atom_isring =  None,
        protein_atom_isO    =  None,
        protein_atom_isN    =  None,


        cross_lig_isring_flag = None,
        cross_lig_isO_flag = None,
        cross_lig_isN_flag = None,

        cross_pro_isring_flag = None,
        cross_pro_isO_flag = None,
        cross_pro_isN_flag = None,


        cross_ligand    = None,
        cross_protein   = None,
        cross_distance  = None,

        cross_bond_index = None, 
        cross_bond_type = None, 
        cross_bond_index_reverse = None, 
        cross_bond_type_reverse = None,

        protein_coords_predict = None,

        complex_mol = None,

        protein_element_batch = None,
        protein_link_t_batch = None,
        protein_link_t_reverse_batch = None,

        ligand_element_batch = None,

        rd_pos = None,
        
        ligand_emb = None,
        pocket_emb = None,



        rate = 10.0,
        scale = True,

    ) -> Tuple[Tensor, Tensor]:
        """Runs one step of the consistency training algorithm.

        Parameters
        ----------
        online_model : nn.Module
            Model that is being trained.
        ema_model : nn.Module
            An EMA of the online model.
        x : Tensor
            Clean data.
        current_training_step : int
            Current step in the training loop.
        total_training_steps : int
            Total number of steps in the training loop.
        **kwargs : Any
            Additional keyword arguments to be passed to the models.

        Returns
        -------
        (Tensor, Tensor)
            The predicted and target values for computing the loss.
        """



        self.num_classes = ligand_atom_feature_dim

        self.loss_v_weight = config.loss_v_weight
        self.loss_exp_weight = config.loss_exp_weight
        self.use_classifier_guide = config.use_classifier_guide

        # atom type diffusion schedule in log space
        alphas_v                = cosine_beta_schedule(self.final_timesteps, config.v_beta_s)
        log_alphas_v            = np.log(alphas_v)
        log_alphas_cumprod_v    = np.cumsum(log_alphas_v)
        self.log_alphas_v       = to_torch_const(log_alphas_v).cuda()
        self.log_one_minus_alphas_v         = to_torch_const(log_1_min_a(log_alphas_v)).cuda()
        self.log_alphas_cumprod_v           = to_torch_const(log_alphas_cumprod_v).cuda()
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v)).cuda()


        # center pos
        center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        _, protein_coords_predict,_ = center_pos(protein_pos, protein_coords_predict, batch_protein, batch_ligand, mode=center_pos_mode) 

        #rdkit 
        _, rd_pos, _ = center_pos(protein_pos, rd_pos.float(), batch_protein, batch_ligand, mode=center_pos_mode) #

        #，，pos，pos
        protein_pos, ligand_pos, offset = center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode) #




        origin_ligand_pos = copy.deepcopy(ligand_pos) #KNN，，KNN
        origin_protein_pos = copy.deepcopy(protein_pos) #KNN，，KNN


        #2
        cross_ligand    = cross_ligand - offset[batch_ligand]
        cross_protein   = cross_protein - offset[batch_protein]
        
        
        #self.final_timesteps，150，，25/15，，num_timesteps[self.initial_timesteps,self.final_timesteps]，[2,25]
        #，：self.final_timesteps = total_training_steps
        #self.final_timesteps = total_training_steps

        #，(>=0), [self.initial_timesteps, self.final_timesteps]，，step=1
        
        if GP.ema_exit:
            num_timesteps = timesteps_schedule(
                current_training_step,
                total_training_steps,
                self.initial_timesteps,
                self.final_timesteps,
            )
        else:
            num_timesteps = improved_timesteps_schedule(
            current_training_step,
            total_training_steps,
            self.initial_timesteps,
            self.final_timesteps,
        )

        #print ('*'*80)
        #print (num_timesteps)  #num_timesteps
        #。，。DDPM
        #
        ''''
        sigmas(step = 25) = 
        tensor([2.0000e-03, 5.2449e-03, 1.2238e-02, 2.6055e-02, 5.1532e-02, 9.5931e-02,
        1.6975e-01, 2.8773e-01, 4.6998e-01, 7.4338e-01, 1.1431e+00, 1.7146e+00,
        2.5152e+00, 3.6171e+00, 5.1092e+00, 7.1005e+00, 9.7232e+00, 1.3136e+01,
        1.7528e+01, 2.3123e+01, 3.0183e+01, 3.9017e+01, 4.9979e+01, 6.3483e+01,
        8.0000e+01])
        】
        sigmas080，080

        '''
        num_graphs = max(batch_ligand) + 1
        sigmas = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, ligand_pos.device
        )

        #
        #sigmas[0] += 1e-8 #，，，

        #print (sigmas,sigmas.shape) # len(sigmas) == num_timesteps
        noise = torch.randn_like(ligand_pos)
        #print (xyzs.shape[0])
        if GP.ema_exit:
            timesteps = torch.randint(0, num_timesteps - 1, (num_graphs,), device=ligand_pos.device)
        else:
            timesteps = lognormal_timestep_distribution(num_graphs, sigmas, self.lognormal_mean, self.lognormal_std) 
            #，,t，

        #print (noise.shape,timesteps.shape)
        current_sigmas = sigmas[timesteps]  #, 
        next_sigmas = sigmas[timesteps + 1] #
        #print (current_sigmas.shape,next_sigmas.shape)

        #，，1，

        current_pos  = current_sigmas.index_select(0, batch_ligand).view([-1, 1])
        next_pos     = next_sigmas.index_select(0, batch_ligand).view([-1, 1]) 

        next_xyzs = ligand_pos + next_pos * noise #sigmas，pad_dims_like(next_sigmas, xyzs) * noise  == 0, next_xyzsxyzs
        #print (next_xyzs.shape

        num_classes = ligand_atom_feature_dim
        #
        log_ligand_v0 = index_to_log_onehot(ligand_v, num_classes) #one-hot,one-hot

        #log_ligand_v0 = F.one_hot(ligand_v, num_classes) #one-hot

        #next_ligand_v_perturbed, next_log_ligand_vt = self.q_v_sample(log_ligand_v0, timesteps + 1, batch_ligand) #
        next_ligand_v_perturbed = log_ligand_v0 #，，
        next_log_ligand_vt = 0


        next_preds = model_forward_wrapper(
            online_model,
            None,
            None,
            next_xyzs, #
            None,
            next_sigmas, #
            self.sigma_data,
            self.sigma_min,

            protein_pos=protein_pos, #
            protein_v=protein_v, 
            batch_protein=batch_protein,

            init_ligand_pos=next_xyzs, # #
            init_ligand_v=next_ligand_v_perturbed,  #
            batch_ligand=batch_ligand,
            time_step=timesteps  + 1 + 1, #1，1，，0， #
            org_ligand_pos = origin_ligand_pos,
            org_protein_pos = origin_protein_pos,
            ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
            protein_element = protein_element, ligand_element = ligand_element,
            scale = scale,
            rate = rate,

            ligand_atom_isring  = ligand_atom_isring,
            ligand_atom_isO     = ligand_atom_isO,
            ligand_atom_isN     = ligand_atom_isN,

            protein_atom_isring = protein_atom_isring,
            protein_atom_isO    = protein_atom_isO,
            protein_atom_isN    = protein_atom_isN,


            cross_lig_isring_flag   = cross_lig_isring_flag,
            cross_lig_isO_flag      = cross_lig_isO_flag,
            cross_lig_isN_flag      = cross_lig_isN_flag,

            cross_pro_isring_flag   = cross_pro_isring_flag,
            cross_pro_isO_flag      = cross_pro_isO_flag,
            cross_pro_isN_flag      = cross_pro_isN_flag,

            cross_ligand    = cross_ligand,
            cross_protein   = cross_protein,
            cross_distance  = cross_distance,

            cross_bond_index = cross_bond_index, 
            cross_bond_type  = cross_bond_type, 
            cross_bond_index_reverse = cross_bond_index_reverse, 
            cross_bond_type_reverse  = cross_bond_type_reverse,

            protein_coords_predict = protein_coords_predict,

            complex_mol = complex_mol,

            protein_element_batch = protein_element_batch,
            protein_link_t_batch = protein_link_t_batch,
            protein_link_t_reverse_batch = protein_link_t_reverse_batch,
            ligand_element_batch = ligand_element_batch,

            rd_pos = rd_pos,
            
        )

        if not GP.ema_exit:
            ema_model = online_model

        with torch.no_grad(): #，，
            current_xyzs = ligand_pos + current_pos * noise

            #current_ligand_v_perturbed, current_log_ligand_vt = self.q_v_sample(log_ligand_v0, timesteps, batch_ligand) #
            current_ligand_v_perturbed = log_ligand_v0
            current_log_ligand_vt = 0

            current_preds = model_forward_wrapper(
                ema_model,
                None,
                None,
                current_xyzs, #
                None,
                current_sigmas, #
                self.sigma_data,
                self.sigma_min,

                protein_pos=protein_pos, #
                protein_v=protein_v, 
                batch_protein=batch_protein,

                init_ligand_pos=current_xyzs, # #
                init_ligand_v=current_ligand_v_perturbed,  #
                batch_ligand=batch_ligand,
                time_step=timesteps + 1, #1，1，，0，，  #
                org_ligand_pos = origin_ligand_pos,
                org_protein_pos = origin_protein_pos,
                ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
                protein_element = protein_element, ligand_element = ligand_element,
                scale = scale,
                rate = rate,

                ligand_atom_isring  = ligand_atom_isring,
                ligand_atom_isO     = ligand_atom_isO,
                ligand_atom_isN     = ligand_atom_isN,

                protein_atom_isring = protein_atom_isring,
                protein_atom_isO    = protein_atom_isO,
                protein_atom_isN    = protein_atom_isN,


                cross_lig_isring_flag   = cross_lig_isring_flag,
                cross_lig_isO_flag      = cross_lig_isO_flag,
                cross_lig_isN_flag      = cross_lig_isN_flag,

                cross_pro_isring_flag   = cross_pro_isring_flag,
                cross_pro_isO_flag      = cross_pro_isO_flag,
                cross_pro_isN_flag      = cross_pro_isN_flag,

                cross_ligand    = cross_ligand,
                cross_protein   = cross_protein,
                cross_distance  = cross_distance,

                cross_bond_index = cross_bond_index, 
                cross_bond_type  = cross_bond_type, 
                cross_bond_index_reverse = cross_bond_index_reverse, 
                cross_bond_type_reverse  = cross_bond_type_reverse,

                protein_coords_predict = protein_coords_predict,

                complex_mol = complex_mol,

                protein_element_batch = protein_element_batch,
                protein_link_t_batch = protein_link_t_batch,
                protein_link_t_reverse_batch = protein_link_t_reverse_batch,
                ligand_element_batch = ligand_element_batch,

                rd_pos = rd_pos,
                
            )
        

        loss_weights = pad_dims_like(improved_loss_weighting(sigmas)[timesteps], next_xyzs)
        
        #st = time.perf_counter()
        
        #，，，amr rmsd

        #rmsd = py_rmsd(next_preds['pred_ligand_pos'], current_preds['pred_ligand_pos'])

        #rmsd = py_rmsd(next_preds['pred_ligand_pos'], origin_ligand_pos)

        n_rmsd = py_rmsd(next_preds['pred_ligand_pos'], origin_ligand_pos) #rmsd，。
        c_rmsd = py_rmsd(current_preds['pred_ligand_pos'], origin_ligand_pos) 
        rmsd   = F.mse_loss(n_rmsd, c_rmsd) #，mse

        #，，，rmsd？ema。
        #，onlineemamse，mse？，，onlineema，
        #，。


        #, ，1，，
        ref_loss   = F.mse_loss(next_preds['pred_ligand_pos'], origin_ligand_pos)

        
        
        '''
        assert ligand_bond_index.shape[0] == 2
        next_pos                = next_preds['pred_ligand_pos']
        next_bond_lengths       = (next_pos[ligand_bond_index[0]] - next_pos[ligand_bond_index[1]]).norm(dim=-1).unsqueeze(-1) # E * 1

        current_pos             = current_preds['pred_ligand_pos']
        current_bond_lengths    = (current_pos[ligand_bond_index[0]] - current_pos[ligand_bond_index[1]]).norm(dim=-1).unsqueeze(-1) # E * 1

        #loss_bond = scatter_mean(((next_bond_lengths - current_bond_lengths) ** 2).sum(-1), ligand_bond_type_batch, dim=0).
        #loss_bond = torch.mean(loss_bond)

        #scatter_mean，,
        #print('next_bond_lengths:', next_bond_lengths.shape)
        #print("next_preds['pred_ligand_pos']:", next_preds['pred_ligand_pos'].shape)
        #next_bond_lengths: torch.Size([888, 1])
        #next_preds['pred_ligand_pos']: torch.Size([422, 3]) 
        loss_bond = F.mse_loss(next_bond_lengths, current_bond_lengths)
        
        
        #dismat，，,list，GPU tensor
        
        n_dismat_list = []
        c_dismat_list = []
        for ids in range(num_graphs): #
            n_pos, c_pos = next_preds['pred_ligand_pos'][batch_ligand == ids], current_preds['pred_ligand_pos'][batch_ligand == ids]
            n_dis = distance(n_pos) #n*n，，
            c_dis = distance(c_pos)
            n_dismat_list.append(n_dis)
            c_dismat_list.append(c_dis)


        n_dismats = torch.cat(n_dismat_list)
        c_dismats = torch.cat(c_dismat_list)

        loss_dismat = F.mse_loss(n_dismats, c_dismats) #nan,rmsd
        '''

        '''
         zamts
        ligand_fill_coords,
        ligand_fill_zmats,
        ligand_fill_masks,
        '''

        #，
        next_cp_ligand_fill_coords    = ligand_fill_coords.clone()
        current_cp_ligand_fill_coords = ligand_fill_coords.clone()

        #print('next_cp_ligand_fill_coords:', next_cp_ligand_fill_coords.shape) #torch.Size([500, 3])
        #print('ligand_fill_masks:', ligand_fill_masks.shape) #torch.Size([500])
        #print('ligand_fill_zmats:', ligand_fill_zmats.shape)
        #print('ligand_fill_masks:', ligand_fill_masks.shape)
        #print('batch_ligand:', batch_ligand)

        '''
        next_cp_ligand_fill_coords: torch.Size([500, 3])
        ligand_fill_masks: torch.Size([500])
        ligand_fill_zmats: torch.Size([500, 4])
        ligand_fill_masks: torch.Size([500])
        '''
        
        #print('next_cp_ligand_fill_coords[ligand_fill_masks==True]:', next_cp_ligand_fill_coords[ligand_fill_masks==True].shape)
        #print("next_preds['pred_ligand_pos'][ligand_fill_atom_order]:", next_preds['pred_ligand_pos'][ligand_fill_atom_order].shape)
        next_cp_ligand_fill_coords[ligand_fill_masks==True]      = next_preds['pred_ligand_pos'][ligand_fill_atom_order] #zmats，
        current_cp_ligand_fill_coords[ligand_fill_masks==True]   = current_preds['pred_ligand_pos'][ligand_fill_atom_order]

        #pred,target,zmats,gmasks: torch.Size([900, 9, 3]) torch.Size([900, 9, 3]) torch.Size([900, 9, 5]) torch.Size([900, 9])
        #
        loss_dismat,loss_bond,loss_angle,loss_dihedral = self.IC_Loss(next_cp_ligand_fill_coords.view(-1,GP.max_atoms,3),\
                                current_cp_ligand_fill_coords.view(-1,GP.max_atoms,3), ligand_fill_zmats.view(-1,GP.max_atoms,5), ligand_fill_masks.view(-1,GP.max_atoms))

        ic_loss=loss_dismat+loss_angle+loss_bond+loss_dihedral
        #exit()
        #end = time.perf_counter()
        #print('a loss_dismat/bond time s:', round(end - st, 4))
        

        #loss_dismat = torch.tensor(0)
        #loss_bond   = torch.tensor(0) #
        #rmsd = torch.tensor(0) #

        #st2 = time.perf_counter()
        #，huber
        #loss_pos = scatter_mean(((current_preds['pred_ligand_pos'] - next_preds['pred_ligand_pos']) ** 2).sum(-1), batch_ligand, dim=0) 
        #F.mse_loss， F.mse_loss，，
        #，。，，，
        #，
        #loss_pos = torch.mean(loss_pos) #

        #loss_pos = F.mse_loss(current_preds['pred_ligand_pos'], next_preds['pred_ligand_pos'])
        #F.mse_loss，，，，，，
        #，msescatter_mean，
        #end2 = time.perf_counter()
        #print('a loss_pos time s:', round(end2 - st2, 4))

        #CMV2
        loss_pos = torch.mean(loss_weights.squeeze() * pseudo_huber_loss(current_preds['pred_ligand_pos'], next_preds['pred_ligand_pos'], batch_ligand)) #loss_weights：


        ####  GP.max_protein_atoms
        c_l_pos = current_preds['pred_ligand_pos']
        c_p_pos = current_preds['final_pos'][current_preds['mask_ligand'] == 0]

        n_l_pos = next_preds['pred_ligand_pos']
        n_p_pos = next_preds['final_pos'][next_preds['mask_ligand'] == 0]

        # 
        #
        fill_c_l_pos_list = []
        mask_c_l_pos_list = []

        fill_c_p_pos_list = []
        mask_c_p_pos_list = []

        for j in range(max(ligand_element_batch) + 1):
            l_mask = batch_ligand[batch_ligand == j]
            p_mask = batch_protein[batch_protein == j]
            new_c_l_pos = c_l_pos[l_mask]  #IndexError: The shape of the mask [224] at index 0 does not match the shape of the indexed tensor [131, 3] at index 0
            new_c_p_pos = c_p_pos[p_mask]  

            mask_c_l_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_c_l_pos[:new_c_l_pos.shape[0]] = True
            fill_c_l_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_c_l_pos[mask_c_l_pos] = new_c_l_pos

            mask_c_p_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_c_p_pos[:new_c_p_pos.shape[0]] = True
            fill_c_p_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_c_p_pos[mask_c_p_pos] = new_c_p_pos

            fill_c_l_pos_list.append(fill_c_l_pos)
            mask_c_l_pos_list.append(mask_c_l_pos)
            fill_c_p_pos_list.append(fill_c_p_pos)
            mask_c_p_pos_list.append(mask_c_p_pos)
        
    
        fill_c_l_pos_s = torch.cat(fill_c_l_pos_list, dim = 0)
        mask_c_l_pos_s = torch.cat(mask_c_l_pos_list, dim = 0)
        fill_c_p_pos_s = torch.cat(fill_c_p_pos_list, dim = 0)
        mask_c_p_pos_s = torch.cat(mask_c_p_pos_list, dim = 0)
        c_lp_distance = self.calculate_distance_matrix_batch(fill_c_l_pos_s.view(-1, GP.max_protein_atoms, 3), fill_c_p_pos_s.view(-1, GP.max_protein_atoms, 3)) # batch_size * n * m

        
        # next
        fill_n_l_pos_list = []
        mask_n_l_pos_list = []

        fill_n_p_pos_list = []
        mask_n_p_pos_list = []

        for j in range(max(ligand_element_batch) + 1):
            #l_mask = next_preds['mask_ligand'][next_preds['batch_all'] == j] == True
            #p_mask = next_preds['mask_ligand'][next_preds['batch_all'] == j] == False
            l_mask = batch_ligand[batch_ligand == j]
            p_mask = batch_protein[batch_protein == j]

            new_n_l_pos = n_l_pos[l_mask] 
            new_n_p_pos = n_p_pos[p_mask]  

            mask_n_l_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_n_l_pos[:new_n_l_pos.shape[0]] = True
            fill_n_l_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_n_l_pos[mask_n_l_pos] = new_n_l_pos

            mask_n_p_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_n_p_pos[:new_n_p_pos.shape[0]] = True
            fill_n_p_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_n_p_pos[mask_n_p_pos] = new_n_p_pos

            fill_n_l_pos_list.append(fill_n_l_pos)
            mask_n_l_pos_list.append(mask_n_l_pos)
            fill_n_p_pos_list.append(fill_n_p_pos)
            mask_n_p_pos_list.append(mask_n_p_pos)
        
    
        fill_n_l_pos_s = torch.cat(fill_n_l_pos_list, dim = 0)
        mask_n_l_pos_s = torch.cat(mask_n_l_pos_list, dim = 0)
        fill_n_p_pos_s = torch.cat(fill_n_p_pos_list, dim = 0)
        mask_n_p_pos_s = torch.cat(mask_n_p_pos_list, dim = 0)
        n_lp_distance = self.calculate_distance_matrix_batch(fill_n_l_pos_s.view(-1, GP.max_protein_atoms, 3), fill_n_p_pos_s.view(-1, GP.max_protein_atoms, 3)) # batch_size * n * m


        #
        #assert c_lp_distance.requires_grad #
        assert n_lp_distance.requires_grad
        assert torch.allclose(c_p_pos, n_p_pos, atol=0.02)
        assert torch.allclose(c_p_pos, origin_protein_pos, atol=0.02)

        #，8ai，
        #true_lp_distance  = self.calculate_distance_matrix(origin_ligand_pos, origin_protein_pos)

        fill_n_origin_ligand_pos_list = []
        mask_n_origin_ligand_pos_list = []

        fill_n_origin_protein_pos_list = []
        mask_n_origin_protein_pos_list = []

        for j in range(max(ligand_element_batch) + 1):
            #l_mask = next_preds['mask_ligand'][next_preds['batch_all'] == j] == True
            #p_mask = next_preds['mask_ligand'][next_preds['batch_all'] == j] == False
            l_mask = batch_ligand[batch_ligand == j]
            p_mask = batch_protein[batch_protein == j]
        
            new_n_origin_ligand_pos = origin_ligand_pos[l_mask] 
            new_n_origin_protein_pos = origin_protein_pos[p_mask]  

            mask_n_origin_ligand_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_n_origin_ligand_pos[:new_n_origin_ligand_pos.shape[0]] = True
            fill_n_origin_ligand_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_n_origin_ligand_pos[mask_n_origin_ligand_pos] = new_n_origin_ligand_pos

            mask_n_origin_protein_pos = torch.zeros([GP.max_protein_atoms], dtype=bool).cuda()
            mask_n_origin_protein_pos[:new_n_origin_protein_pos.shape[0]] = True
            fill_n_origin_protein_pos = torch.zeros([GP.max_protein_atoms, 3]).cuda()
            fill_n_origin_protein_pos[mask_n_origin_protein_pos] = new_n_origin_protein_pos

            fill_n_origin_ligand_pos_list.append(fill_n_origin_ligand_pos)
            mask_n_origin_ligand_pos_list.append(mask_n_origin_ligand_pos)
            fill_n_origin_protein_pos_list.append(fill_n_origin_protein_pos)
            mask_n_origin_protein_pos_list.append(mask_n_origin_protein_pos)
        
    
        fill_n_origin_ligand_pos_s = torch.cat(fill_n_origin_ligand_pos_list, dim = 0)
        mask_n_origin_ligand_pos_s = torch.cat(mask_n_origin_ligand_pos_list, dim = 0)
        fill_n_origin_protein_pos_s = torch.cat(fill_n_origin_protein_pos_list, dim = 0)
        mask_n_origin_protein_pos_s = torch.cat(mask_n_origin_protein_pos_list, dim = 0)
        true_lp_distance = self.calculate_distance_matrix_batch(fill_n_origin_ligand_pos_s.view(-1, GP.max_protein_atoms, 3), fill_n_origin_protein_pos_s.view(-1, GP.max_protein_atoms, 3)) # batch_size * n * m

        mask_cutoff = true_lp_distance < 8
        cross_distance_loss = F.mse_loss(c_lp_distance[mask_cutoff], n_lp_distance[mask_cutoff]) #，.
        

        #mse(, )
        ref_d  = true_lp_distance
        pred_d = n_lp_distance
        ref_cross_distance_loss = F.mse_loss(ref_d[mask_cutoff], pred_d[mask_cutoff]) #

        #assert ref_d.requires_grad
        assert pred_d.requires_grad
        


        '''
        #, inf，，，inf0，0，
        c_ll_distance = self.calculate_distance_matrix(c_l_pos, c_l_pos) # n * n
        n_ll_distance = self.calculate_distance_matrix(n_l_pos, n_l_pos) # n * n

        #  torch.where  inf  -inf  0
        c_ll_distance = torch.where(torch.nan(c_ll_distance), torch.tensor(0.0, requires_grad=True), c_ll_distance)
        n_ll_distance = torch.where(torch.nan(n_ll_distance), torch.tensor(0.0, requires_grad=True), n_ll_distance)

        true_ll_distance     = self.calculate_distance_matrix(origin_ligand_pos, origin_ligand_pos)
        mask_cutoff          = true_ll_distance < 8
        ligand_distance_loss = F.mse_loss(c_ll_distance[mask_cutoff], n_ll_distance[mask_cutoff])
        '''
        ligand_distance_loss = 0.0



        #exit()

        '''
        # atom type loss，
        log_ligand_v_recon = F.log_softmax(next_preds['pred_ligand_v'], dim=-1) #
        log_v_model_prob   = self.q_v_posterior(log_ligand_v_recon, next_log_ligand_vt, timesteps, batch_ligand) #
        log_v_true_prob    = self.q_v_posterior(log_ligand_v0, next_log_ligand_vt, timesteps, batch_ligand) #
        

        #KL
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                log_v_true_prob=log_v_true_prob, t=timesteps, batch=batch_ligand)
        loss_v = torch.mean(kl_v)
        '''

        loss_v = torch.tensor(0)
        

        #
        #loss_exp = F.mse_loss(next_preds['final_exp_pred'], affinity) #,
        loss_exp = torch.tensor(0)

    

        #loss_exp = torch.tensor(0)
        
        #rmsd
        #
        if self.use_classifier_guide:  #The default is True
            #loss = ic_loss * GP.loss_weight['ic'] + loss_pos * GP.loss_weight['xyz']
            loss = ic_loss * GP.loss_weight['ic'] + loss_pos * GP.loss_weight['xyz'] + cross_distance_loss * GP.loss_weight['cross_distance'] + ref_cross_distance_loss * GP.loss_weight['ref_cross'] + ref_loss * GP.loss_weight['ref']
        else:
            #loss = ic_loss * GP.loss_weight['ic'] + loss_pos * GP.loss_weight['xyz']
            loss = ic_loss * GP.loss_weight['ic'] + loss_pos * GP.loss_weight['xyz'] + cross_distance_loss * GP.loss_weight['cross_distance'] + ref_cross_distance_loss * GP.loss_weight['ref_cross'] + ref_loss * GP.loss_weight['ref']

        

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_exp': loss_exp,
            'loss': loss,
            'rmsd': rmsd, 
            'loss_dismat': loss_dismat,
            'loss_bond': loss_bond,
            'loss_angle': loss_angle,
            'loss_dihedral': loss_dihedral,
            'x0': ligand_pos,  #
            'pred_ligand_pos': next_preds['pred_ligand_pos'],
            'pred_ligand_v': torch.zeros_like(log_ligand_v0).cuda(),
            'pred_exp': next_preds['final_exp_pred'], #
            'pred_pos_noise': torch.zeros_like(next_preds['pred_ligand_pos']).cuda(),
            'ligand_v_recon': torch.zeros_like(log_ligand_v0).cuda(), #
            'final_ligand_h': next_preds['final_ligand_h']  #
        }

class ConsistencySamplingAndEditing:
    """Implements the Consistency Sampling and Zero-Shot Editing algorithms.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_data: float = 0.5) -> None:
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data



    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_prob = log_sample_categorical(log_qvt_v0)
        sample_index = sample_prob.argmax(dim=-1) #，
        log_sample = index_to_log_onehot(sample_index, self.num_classes) #one-hot，
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, v0) * q(vt-1 | v0) / q(vt | v0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0
    


    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def __call__(
        self,

        model: nn.Module,
        protein_atom_feature_dim,
        ligand_atom_feature_dim,

        config,

        #ground truth
        #protein_pos,
        #protein_v,
        affinity,
        #batch_protein,
        ligand_pos,
        ligand_v,
        org_ligand_pos,
        #batch_ligand,

        #sample params
        guide_mode,
        value_model,
        type_grad_weight,
        pos_grad_weight,

        protein_pos,
        protein_v,
        batch_protein,

        init_ligand_pos,
        init_ligand_v,
        batch_ligand,

        num_steps,
        center_pos_mode,

        ligand_bond_index, ligand_bond_type, ligand_bond_type_batch,


        ligand_atom_isring  =  None,
        ligand_atom_isO     =  None,
        ligand_atom_isN     =  None,

        protein_atom_isring =  None,
        protein_atom_isO    =  None,
        protein_atom_isN    =  None,



        cross_lig_isring_flag = None,
        cross_lig_isO_flag = None,
        cross_lig_isN_flag = None,

        cross_pro_isring_flag = None,
        cross_pro_isO_flag = None,
        cross_pro_isN_flag = None,


        cross_ligand    = None,
        cross_protein   = None,
        cross_distance  = None,


        cross_bond_index = None, 
        cross_bond_type = None, 
        cross_bond_index_reverse = None, 
        cross_bond_type_reverse = None,

        protein_coords_predict = None,

        complex_mol = None,

        protein_element_batch = None,
        protein_link_t_batch = None,
        protein_link_t_reverse_batch = None,

        ligand_element_batch = None,


        protein_element = None,
        ligand_element  = None,

        rd_pos = None,


        batch_center_pos = None,

        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        initial_timesteps: int = 2, #2
        final_timesteps: int = 25,  ##self.final_timesteps，150，，25/15，，，):
        total_training_steps: int = 25,


        #not use
        feats: Tensor = None,
        adjs: Tensor = None,
        y: Tensor = None,
        gmasks: Tensor = None,
        sigmas: Iterable[Union[Tensor, float]]  = None,


        mask: Optional[Tensor] = None,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        start_from_y: bool = False,
        add_initial_noise: bool = False, # default True
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Runs the sampling/zero-shot editing loop.

        With the default parameters the function performs consistency sampling.

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        y : Tensor
            Reference sample e.g: a masked image or noise.
        sigmas : Iterable[Union[Tensor, float]]
            Decreasing standard deviations of the noise.
        mask : Tensor, default=None
            A mask of zeros and ones with ones indicating where to edit. By
            default the whole sample will be edited. This is useful for sampling.
        transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            An invertible linear transformation. Defaults to the identity function.
        inverse_transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            Inverse of the linear transformation. Defaults to the identity function.
        start_from_y : bool, default=False
            Whether to use y as an initial sample and add noise to it instead of starting
            from random gaussian noise. This is useful for tasks like style transfer.
        add_initial_noise : bool, default=True
            Whether to add noise at the start of the schedule. Useful for tasks like interpolation
            where noise will alerady be added in advance.
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1] range.
        verbose : bool, default=False
            Whether to display the progress bar.
        **kwargs : Any
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        Tensor
            Edited/sampled sample.
        """

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps
        self.num_classes = ligand_atom_feature_dim
        self.ref_atom_type = copy.deepcopy(ligand_v)


        # atom type diffusion schedule in log space
        alphas_v                = cosine_beta_schedule(self.final_timesteps, 0.01)
        log_alphas_v            = np.log(alphas_v)
        log_alphas_cumprod_v    = np.cumsum(log_alphas_v)
        self.log_alphas_v       = to_torch_const(log_alphas_v).cuda()
        self.log_one_minus_alphas_v         = to_torch_const(log_1_min_a(log_alphas_v)).cuda()
        self.log_alphas_cumprod_v           = to_torch_const(log_alphas_cumprod_v).cuda()
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v)).cuda()


        num_graphs = max(batch_ligand) + 1


        #，sigma，sigma，,sigma
        #self.final_timesteps[1, *]
        sigmas = karras_schedule(
                self.final_timesteps, self.sigma_min, self.sigma_max, self.rho, init_ligand_pos.device
            )

        #print('sigmas1:', sigmas)
        if self.final_timesteps == 1:
            sigmas= reversed(sigmas) #，
            sigmas[-1] = 80 # sigmas[-1] += 1e-8。1，sigmas1: tensor([0.0020]), ，c_cout0, 
            #，，80。2，
            #sigmas[-1] += 1e-8
        else:
            #sigmas= reversed(sigmas)[:-1] #，，，, 1，
            sigmas= reversed(sigmas)
            #print('sigmas2:', sigmas[-1].detach().cpu().numpy())
            sigmas[-1] += 1e-8  #,，c_out，，0
            #print('sigmas3:', sigmas[-1].detach().cpu().numpy())
            #sigmas2: 0.0019999996
            #sigmas3: 0.0020000096

        

        time_list = np.array(list(reversed(list(range(len(sigmas)))))) #，1。0，





        # Set mask to all ones which is useful for sampling and style transfer
        if mask is None:
            mask = torch.ones_like(y)

        # Use y as an initial sample which is useful for tasks like style transfer
        # and interpolation where we want to use content from the reference sample
        x = y if start_from_y else torch.zeros_like(y)

        # Sample at the end of the schedule
        y = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)
        # For tasks like interpolation where noise will already be added in advance we
        # can skip the noising process

        #print('sigmas[0]:', sigmas[0])
        x = y + batch_center_pos + sigmas[0] * torch.randn_like(y)
        #x = y + batch_center_pos + torch.randn_like(y)
        #x = y + torch.randn_like(y)

        #x = y + org_ligand_pos

        #x = y + sigmas[0] * torch.randn_like(y)

        #x = y + batch_center_pos + sigmas[0] * org_ligand_pos #

        #x = org_ligand_pos + sigmas[0] * torch.randn_like(y) #

        #x = y + batch_center_pos  + org_ligand_pos  

        #print('origin_cross_ligand:', cross_ligand.shape)
        #print('org_ligand_pos:', org_ligand_pos.shape)
        #print('x:', x.shape)

        #print('origin_cross_ligand[:3]:', cross_ligand[:])
        #print('org_ligand_pos[:3]:', org_ligand_pos[:])
        pos_traj, v_traj, exp_traj, exp_atom_traj = [], [], [], []
        
        #pos_traj.append(x.clone().cpu()) 
        
        init_protein_pos, init_ligand_pos, offset = center_pos(protein_pos, x, batch_protein, batch_ligand, mode=center_pos_mode) #

        #rdkit 
        _, rd_pos, _ = center_pos(protein_pos, rd_pos.float(), batch_protein, batch_ligand, mode=center_pos_mode) #


        org_protein_pos, org_ligand_pos, org_offset = center_pos(protein_pos, org_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode) #
        

        _, protein_coords_predict, _ = center_pos(protein_pos, protein_coords_predict, batch_protein, batch_ligand, mode=center_pos_mode)


        #print('origin_cross_ligand[:3]:', cross_ligand[:3])
        #print('org_ligand_pos[:3]:', org_ligand_pos[:3]) #，，，x，, org_ligand_posrdkit，

        #2
        cross_ligand    = cross_ligand - offset[batch_ligand]
        cross_protein   = cross_protein - offset[batch_protein]


        
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        protein_pos = init_protein_pos

        '''
        np.set_printoptions(suppress=True, precision=4)
        torch.set_printoptions(sci_mode=False, precision=4)

        print('cross_ligand:', cross_ligand.shape)
        print('org_ligand_pos:', org_ligand_pos.shape)
        print('ligand_pos:', ligand_pos.shape)

        print('cross_ligand[:3]:', cross_ligand[:3])
        print('org_ligand_pos[:3]:', org_ligand_pos[:3])
        print('ligand_pos[:3]:', ligand_pos[:3])

        #，，？x
        print('cross_protein[:3]:', cross_protein[:])
        print('protein_pos[:3]:', protein_pos[:])

        print('cross_protein.shape:', cross_protein.shape)
        print('protein_pos.shape:', protein_pos.shape)

        raise Exception('stop')
        '''



        sigma = torch.full((num_graphs,), sigmas[0], dtype=x.dtype, device=x.device)
        #sigma，step
        timesteps = torch.full((num_graphs,), time_list[0], dtype=torch.int64, device=init_ligand_pos.device)


        with torch.no_grad():
        #with torch.enable_grad():
            preds, type_grad, pos_grad = self.consistency_pv_joint_guide(
                model,
                None,
                None,
                ligand_pos, #
                None,
                sigma, #
                self.sigma_data,
                self.sigma_min,

                protein_pos=protein_pos, #
                protein_v=protein_v, 
                batch_protein=batch_protein,

                init_ligand_pos=ligand_pos, # #
                init_ligand_v=ligand_v,  
                batch_ligand=batch_ligand,
                time_step=timesteps + 1, #
                org_ligand_pos = org_ligand_pos,
                org_protein_pos = org_protein_pos,
                ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
                args = None,
                protein_element = protein_element,
                ligand_element  = ligand_element,
                sample = True,   #，，sampleTrue, 
                scale = True, #
                
                ligand_atom_isring  = ligand_atom_isring,
                ligand_atom_isO     = ligand_atom_isO,
                ligand_atom_isN     = ligand_atom_isN,

                protein_atom_isring = protein_atom_isring,
                protein_atom_isO    = protein_atom_isO,
                protein_atom_isN    = protein_atom_isN,

                cross_lig_isring_flag   = cross_lig_isring_flag,
                cross_lig_isO_flag      = cross_lig_isO_flag,
                cross_lig_isN_flag      = cross_lig_isN_flag,

                cross_pro_isring_flag   = cross_pro_isring_flag,
                cross_pro_isO_flag      = cross_pro_isO_flag,
                cross_pro_isN_flag      = cross_pro_isN_flag,

                cross_ligand    = cross_ligand,
                cross_protein   = cross_protein,
                cross_distance  = cross_distance,

                cross_bond_index = cross_bond_index, 
                cross_bond_type = cross_bond_type, 
                cross_bond_index_reverse = cross_bond_index_reverse, 
                cross_bond_type_reverse = cross_bond_type_reverse,

                protein_coords_predict = protein_coords_predict,

                complex_mol = complex_mol,

                protein_element_batch = protein_element_batch,
                protein_link_t_batch = protein_link_t_batch,
                protein_link_t_reverse_batch = protein_link_t_reverse_batch,

                ligand_element_batch = ligand_element_batch,

                rd_pos = rd_pos,



            )


        if clip_denoised:
            preds['pred_ligand_pos'] = preds['pred_ligand_pos'].clamp(min=-1.0, max=1.0)
        
        ligand_pos = preds['pred_ligand_pos']

        #
        #print('ligand_pos:', ligand_pos.requires_grad) #True
        #print('protein_pos:', protein_pos.requires_grad)
        #print('cross_distance[0]:', cross_distance[0].requires_grad)
        #ligand_pos = self.force_gradient(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach())

        #ligand_pos = ligand_pos + self.Distance_Opt(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach())

        batch_all   = preds['batch_all']
        mask_ligand = preds['mask_ligand']
        final_pos   = preds['final_pos']
        p_pos = [final_pos[batch_all == k][mask_ligand[batch_all == k] == 0] for k in range(max(batch_all) + 1)]
        #assert np.allclose(p_pos[0].cpu(), p_pos[-1].cpu(), atol=0.02) #
        
        if GP.final_timesteps == 1:
            if GP.with_MMFF_guide:
                #：
                '''
                x, #
                rdkit_mols, #ridkt mol,, 
                gmasks.bool(), #
                loop=guide_loops, #，1，
                show_state=show_state,
                min_type=min_type,  #，LBFGS
                fix_masks=fix_mask, #，
                pocket_masks=pocket_labels.bool(), #
                ligand_masks=ligand_labels.bool()  #
                '''
                preds['final_pos'][preds['mask_ligand']] = preds['pred_ligand_pos']
                complex_pos = preds['final_pos']
                

                with torch.enable_grad():
                    #_,  cross_loss = self.Distance_Opt(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach(), min_type = GP.min_type)
                    if GP.guide_type=='synchronous': #，x，，，
                        if GP.opt_types=="complex":
                            pass
                            #x_moves,energy_min=opt_complex_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=1,show_state=Ture,min_type=LBFGS,fix_masks=fix_mask,pocket_masks=pocket_labels.bool(),ligand_masks=ligand_labels.bool())
                        else:
                            pass
                            #x_moves,energy_min=opt_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask)
                    
                    #，，。+complex, asynchronous 
                    else:
                        try:
                            if GP.opt_types=="complex":
                                x_moves,energy_min=opt_complex_coords_moves(complex_pos.clone().detach(), copy.deepcopy(complex_mol), preds['batch_all'].clone().detach(), loop=1, show_state=True, min_type=GP.min_type, mask_ligand = preds['mask_ligand'].clone().detach(), cross_loss = None, ligand_pos = ligand_pos.clone().detach(), protein_pos = protein_pos.clone().detach(), cross_distance = copy.deepcopy(cross_distance))
                            else:
                                x_moves,energy_min=opt_coords_moves(complex_pos.clone().detach(), copy.deepcopy(complex_mol), preds['batch_all'].clone().detach(), loop=1, show_state=True, min_type=GP.min_type, mask_ligand = preds['mask_ligand'].clone().detach(), cross_loss = None, ligand_pos = ligand_pos.clone().detach(), protein_pos = protein_pos.clone().detach(), cross_distance = copy.deepcopy(cross_distance))
                        except Exception as e:
                            print(e)
                            x_moves = 0


                ligand_pos = ligand_pos+x_moves #x_moves，，+

        


        ligand_pos = self.__mask_transform(ligand_pos, y, mask, transform_fn, inverse_transform_fn)
        exp_pred = preds['final_exp_pred']


        ori_ligand_pos = ligand_pos + offset[batch_ligand]  #
        pos_traj.append(ori_ligand_pos.clone().cpu())       #
        v_traj.append(ligand_v.clone().cpu())               #v
    
    



        #
        if exp_pred is not None:
            exp_traj.append(exp_pred.clone().cpu())
            exp_atom_traj.append(preds['atom_affinity'].clone().cpu())


        for step, (sigma, t) in enumerate(zip(sigmas[1:], time_list[1:])):
            #print('sigma:', sigma)
            noise = torch.randn_like(ligand_pos)

            #，time_step
            #timesteps = torch.randint(0, num_timesteps - 1, (ligand_pos.shape[0],), device=ligand_pos.device) #[ ),
            #timesteps = torch.full((num_graphs,), t, dtype=ligand_pos.dtype, device=ligand_pos.device)
            timesteps = torch.full((num_graphs,), t, dtype=torch.int64, device=ligand_pos.device)
            #print (noise.shape,timesteps.shape)
            next_sigma = torch.full((num_graphs,), sigma, dtype=ligand_pos.dtype, device=ligand_pos.device) #，，sigma，sigma,
            #print (current_sigmas.shape,next_sigmas.shape)
            #print(f'step{step+1}:', next_sigma)

            new_sigma = (next_sigma**2 - self.sigma_min**2) ** 0.5
            #print(f'step{step + 1} new_sigma:', new_sigma)
            next_pos  = new_sigma.index_select(0, batch_ligand).view([-1, 1])

                
            #
            ligand_pos = ligand_pos + next_pos * noise 


            #sigmas，pad_dims_like(next_sigmas, xyzs) * noise  == 0, next_xyzsxyzs
            #sigma**2 - self.sigma_min**2，simga，0，，x


            with torch.no_grad():
            #with torch.enable_grad():
                preds, type_grad, pos_grad = self.consistency_pv_joint_guide(
                    model,
                    None,
                    None,
                    ligand_pos, #
                    None,
                    next_sigma, #
                    self.sigma_data,
                    self.sigma_min,

                    protein_pos=protein_pos, #
                    protein_v=protein_v, 
                    batch_protein=batch_protein,

                    init_ligand_pos=ligand_pos, # #
                    init_ligand_v=ligand_v,   #
                    batch_ligand=batch_ligand,
                    time_step=timesteps + 1, #
                    org_ligand_pos = org_ligand_pos,
                    org_protein_pos = org_protein_pos,
                    ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
                    args = None,
                    protein_element = protein_element,
                    ligand_element  = ligand_element,
                    sample = True,  #，，sampleTrue, 
                    scale = True, 

                    ligand_atom_isring  = ligand_atom_isring,
                    ligand_atom_isO     = ligand_atom_isO,
                    ligand_atom_isN     = ligand_atom_isN,

                    protein_atom_isring = protein_atom_isring,
                    protein_atom_isO    = protein_atom_isO,
                    protein_atom_isN    = protein_atom_isN,

                    cross_lig_isring_flag   = cross_lig_isring_flag,
                    cross_lig_isO_flag      = cross_lig_isO_flag,
                    cross_lig_isN_flag      = cross_lig_isN_flag,

                    cross_pro_isring_flag   = cross_pro_isring_flag,
                    cross_pro_isO_flag      = cross_pro_isO_flag,
                    cross_pro_isN_flag      = cross_pro_isN_flag,

                    cross_ligand    = cross_ligand,
                    cross_protein   = cross_protein,
                    cross_distance  = cross_distance,

                    cross_bond_index = cross_bond_index, 
                    cross_bond_type = cross_bond_type, 
                    cross_bond_index_reverse = cross_bond_index_reverse, 
                    cross_bond_type_reverse = cross_bond_type_reverse,

                    protein_coords_predict = protein_coords_predict,

                    complex_mol = complex_mol,

                    protein_element_batch = protein_element_batch,
                    protein_link_t_batch = protein_link_t_batch,
                    protein_link_t_reverse_batch = protein_link_t_reverse_batch,

                    ligand_element_batch = ligand_element_batch,

                    rd_pos = rd_pos,
                )
            
            if clip_denoised:
                preds['pred_ligand_pos'] = preds['pred_ligand_pos'].clamp(min=-1.0, max=1.0)

            ligand_pos = preds['pred_ligand_pos']

            #
            #print('ligand_pos:', ligand_pos.requires_grad)
            #print('protein_pos:', protein_pos.requires_grad)
            #print('cross_distance[0]:', cross_distance[0].requires_grad)
            #ligand_pos = self.force_gradient(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach())
            #ligand_pos = ligand_pos + self.Distance_Opt(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach())
            #ligand_pos = self.Distance_Opt(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach())

            batch_all   = preds['batch_all']
            mask_ligand = preds['mask_ligand']
            final_pos   = preds['final_pos']
            p_pos = [final_pos[batch_all == k][mask_ligand[batch_all == k] == 0] for k in range(max(batch_all) + 1)]
            #assert np.allclose(p_pos[0].cpu(), p_pos[-1].cpu(), atol=0.02) #

            if step in list(range(len(sigmas[1:])))[-GP.force_step:]:
                if GP.with_MMFF_guide:
                    #：
                    '''
                    x, #
                    rdkit_mols, #ridkt mol,, 
                    gmasks.bool(), #
                    loop=guide_loops, #，1，
                    show_state=show_state,
                    min_type=min_type,  #，LBFGS
                    fix_masks=fix_mask, #，
                    pocket_masks=pocket_labels.bool(), #
                    ligand_masks=ligand_labels.bool()  #
                    '''
                    preds['final_pos'][preds['mask_ligand']] = preds['pred_ligand_pos']
                    complex_pos = preds['final_pos']
                    
                    with torch.enable_grad():
                        #_,  cross_loss = self.Distance_Opt(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach(), min_type = GP.min_type)
                        if GP.guide_type=='synchronous': #，x，，，
                            if GP.opt_types=="complex":
                                pass
                                #x_moves,energy_min=opt_complex_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=1,show_state=Ture,min_type=LBFGS,fix_masks=fix_mask,pocket_masks=pocket_labels.bool(),ligand_masks=ligand_labels.bool())
                            else:
                                pass
                                #x_moves,energy_min=opt_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask)
                        
                        #，，。+complex, asynchronous 
                        else:
                            try:
                                if GP.opt_types=="complex":
                                    x_moves,energy_min=opt_complex_coords_moves(complex_pos.clone().detach(), copy.deepcopy(complex_mol), preds['batch_all'].clone().detach(), loop=GP.loop, show_state=True, min_type=GP.min_type, mask_ligand = preds['mask_ligand'].clone().detach(), cross_loss = None, ligand_pos = ligand_pos.clone().detach(), protein_pos = protein_pos.clone().detach(), cross_distance = copy.deepcopy(cross_distance))
                                else:
                                    x_moves,energy_min=opt_coords_moves(complex_pos.clone().detach(), copy.deepcopy(complex_mol), preds['batch_all'].clone().detach(), loop=GP.loop, show_state=True, min_type=GP.min_type, mask_ligand = preds['mask_ligand'].clone().detach(), cross_loss = None, ligand_pos = ligand_pos.clone().detach(), protein_pos = protein_pos.clone().detach(), cross_distance = copy.deepcopy(cross_distance))
                            except Exception as e:
                                print(e)
                                x_moves = 0

                    ligand_pos = ligand_pos+x_moves #x_moves，，+# * 0.1, 5step



            ligand_pos = self.__mask_transform(ligand_pos, y, mask, transform_fn, inverse_transform_fn)
            exp_pred = preds['final_exp_pred']



            ori_ligand_pos = ligand_pos + offset[batch_ligand]  #
            pos_traj.append(ori_ligand_pos.clone().cpu())       #
            v_traj.append(ligand_v.clone().cpu())               #v
        
            
            #
            if exp_pred is not None:
                exp_traj.append(exp_pred.clone().cpu())
                exp_atom_traj.append(preds['atom_affinity'].clone().cpu())
            


        ligand_pos = ligand_pos + offset[batch_ligand]

        #sdf，
        coords_predict = protein_coords_predict + offset[batch_ligand]

        assert coords_predict.shape == ligand_pos.shape
                
        #ligand_pos = pos_traj[-1]
        #ligand_v   = v_traj[-1] #
        return {
            'pos': ligand_pos,
            'coords_predict': coords_predict,
            'v': ligand_v,
            'exp': exp_traj[-1] if len(exp_traj) else [],
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'exp_traj': exp_traj,
            'exp_atom_traj': exp_atom_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
        }








    def Distance_Opt_old(self, ligand_pos, protein_pos, cross_distance, iterations=1, min_type='AdamW', early_stoping=10):
        with torch.enable_grad():
            #
            ligand_pos.requires_grad = True
            #coor_pred_detach = coor_pred.detach()
            ligand_pos_Parm = torch.nn.Parameter(ligand_pos.detach().clone()).to(ligand_pos.device)
            #protein_pos.requires_grad = True
            pred_distance = torch.cdist(ligand_pos_Parm,protein_pos,compute_mode='donot_use_mm_for_euclid_dist')
            #print('pred_distance:', pred_distance.requires_grad) # False
            Distance_loss = F.mse_loss(pred_distance, cross_distance)
            print ('before optim distance:', Distance_loss)
            if min_type == 'SGD':
                optimizer = torch.optim.SGD([ligand_pos_Parm], lr=0.02)
            elif min_type =='AdamW':
                optimizer = torch.optim.AdamW([ligand_pos_Parm], lr=0.02)
            else:
                optimizer = torch.optim.LBFGS([ligand_pos_Parm], lr=0.0001) #LBFGS

            for i in range(iterations): 
                def closure():
                    optimizer.zero_grad()
                    pred_distance = torch.cdist(ligand_pos_Parm,protein_pos,compute_mode='donot_use_mm_for_euclid_dist')
                    loss = F.mse_loss(pred_distance, cross_distance)
                    loss.backward()
                    return loss
                close_loss = closure()
                if min_type == 'SGD':
                    optimizer.step()
                elif min_type == 'AdamW':
                    optimizer.step()
                else: 
                    optimizer.step(closure)
                #print('close_loss:', close_loss.item())

            ligand_pos_Min = ligand_pos_Parm.detach().clone().to(ligand_pos.device)
            pred_distance = torch.cdist(ligand_pos_Min, protein_pos, compute_mode='donot_use_mm_for_euclid_dist')
            Distance_loss = F.mse_loss(pred_distance, cross_distance)
            print('after optim ditance:', Distance_loss)

            return ligand_pos_Min, close_loss
        


    def Distance_Opt(self, ligand_pos, protein_pos, cross_distance, iterations=1, min_type='AdamW', early_stoping=10):
        with torch.enable_grad():
            #
            ligand_pos.requires_grad = True
            #coor_pred_detach = coor_pred.detach()
            ligand_pos_Parm = torch.nn.Parameter(ligand_pos.detach().clone()).to(ligand_pos.device)
            #protein_pos.requires_grad = True
            pred_distance = torch.cdist(ligand_pos_Parm,protein_pos,compute_mode='donot_use_mm_for_euclid_dist')
            #print('pred_distance:', pred_distance.requires_grad) # False
            Distance_loss = F.mse_loss(pred_distance, cross_distance)
            print ('before optim distance:', Distance_loss)
            if min_type == 'SGD':
                optimizer = torch.optim.SGD([ligand_pos_Parm], lr=0.02)
            elif min_type =='AdamW':
                optimizer = torch.optim.AdamW([ligand_pos_Parm], lr=0.02)
            else:
                optimizer = torch.optim.LBFGS([ligand_pos_Parm], lr=0.0001) #LBFGS

            for i in range(iterations): 
                def closure():
                    #optimizer.zero_grad()
                    pred_distance = torch.cdist(ligand_pos_Parm,protein_pos,compute_mode='donot_use_mm_for_euclid_dist')
                    loss = F.mse_loss(pred_distance, cross_distance)
                    #loss.backward() #
                    return loss
                close_loss = closure()

                #if min_type == 'SGD':
                    #optimizer.step()
                #elif min_type == 'AdamW':
                    #optimizer.step()
                #else: 
                    #optimizer.step(closure)
                #print('close_loss:', close_loss.item())

            ligand_pos_Min = ligand_pos_Parm.detach().clone().to(ligand_pos.device)
            pred_distance = torch.cdist(ligand_pos_Min, protein_pos, compute_mode='donot_use_mm_for_euclid_dist')
            Distance_loss = F.mse_loss(pred_distance, cross_distance)
            print('after optim ditance:', Distance_loss)

            return ligand_pos_Min, close_loss








    def force_gradient(self, ligand_pos, protein_pos, cross_distance, iterations=1, early_stoping=1):
        gradient_data_dict = {
                    'ligand_pos': ligand_pos,
                    'protein_pos': protein_pos,
                    'cross_distance': cross_distance
                    }
        
        torch.save(gradient_data_dict, 'gradient_data_dict.pt')
        
        #print('ligand_pos:', ligand_pos.requires_grad) # True
        #print('protein_pos:', protein_pos.requires_grad)# False
        #print('cross_distance:', cross_distance.requires_grad)# False
    
        #
        ligand_pos.requires_grad = True
        #coor_pred_detach = coor_pred.detach()
        ligand_pos_Parm = torch.nn.Parameter(ligand_pos.detach().clone()).to(ligand_pos.device)
        #protein_pos.requires_grad = True
        pred_distance = self.calculate_distance_matrix(ligand_pos_Parm, protein_pos) #，ligand_pos
        assert pred_distance.shape == cross_distance.shape
        
        #print('pred_distance:',pred_distance.requires_grad) # False

        distance = F.mse_loss(pred_distance, cross_distance)
        print('befor optim ditance:', distance)
        clone_protein_pos, clone_cross_distance = protein_pos.clone(), cross_distance.clone()

        optimizer = torch.optim.LBFGS([ligand_pos_Parm], lr=1.0)
        bst_loss, times = 10000.0, 0

        pred_distance.requires_grad = True
        cross_distance.requires_grad = True

        #print(tensor_with_grad.requires_grad)
        #print('ligand_pos:', ligand_pos.requires_grad)
        #print('pred_distance_detached:', pred_distance.requires_grad)
        #print('cross_distance_detached:', cross_distance.requires_grad)
            
        for i in range(iterations): 
            def closure():
                optimizer.zero_grad()
                ##，coordsrdkit，，
                #GNN，coords（rdkit，，，rdkit，），
                # ，rdkitmse，rdkit，
                #loss = F.mse_loss(pred_distance.detach().requires_grad_(True), cross_distance.detach().requires_grad_(True))
                loss = F.mse_loss(pred_distance, cross_distance) * 1 #mse，？
                #loss = self.scoring_function(ligand_pos, protein_pos, pred_distance_detached, cross_distance_detached)
                loss.backward(retain_graph=True)
                print('optim loss:', loss.item())
                return loss
            loss = optimizer.step(closure)
            if loss.item() < bst_loss:
                bst_loss = loss.item()
                times = 0 
            else:
                times += 1
                if times > early_stoping:
                    break

        

        assert torch.equal(clone_protein_pos,protein_pos) and torch.equal(clone_cross_distance,cross_distance)
        pred_distance = self.calculate_distance_matrix(ligand_pos_Parm, protein_pos)
        distance = F.mse_loss(pred_distance, cross_distance) #，
        print('after optim ditance:', distance)
        print('-----------------------------------------------------')

        return ligand_pos_Parm.detach()



    def calculate_distance_matrix(self, A, B):
        """
        

        :
        A (torch.Tensor):  (n, 3) 
        B (torch.Tensor):  (m, 3) 

        :
        torch.Tensor:  (n, m) 
        """
        # A  (n, 3)，B  (m, 3)
        #  A  B 
        print('A:', A.requires_grad) # True
        print('B:', B.requires_grad)# True
        print('A.unsqueeze(1):', A.unsqueeze(1).requires_grad)# True
        print('B.unsqueeze(0):', B.unsqueeze(0).requires_grad)# True
        diff = A.unsqueeze(1) - B.unsqueeze(0)  # diff  (n, m, 3),# ，？
        print('diff.shape:', diff.shape)
        #print('diff:', diff)
        print('A + A:', (A + A).requires_grad) # False
        print('diff:', diff.requires_grad) # False
        dist_matrix = torch.sqrt(torch.sum(diff**2, dim=2))  # dist_matrix  (n, m)
        print('dist_matrix:', dist_matrix.requires_grad)
        return dist_matrix


    def consistency_pv_joint_guide(
        self,
        model: nn.Module,
        feats: Tensor = None,
        adjs: Tensor = None,
        xyzs: Tensor = None,
        gmasks: Tensor = None,
        sigma: Tensor = None,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,

        
        protein_pos=None, #
        protein_v=None, 
        batch_protein=None,

        init_ligand_pos=None, #
        init_ligand_v=None,  #
        batch_ligand=None,
        time_step=None,
        org_ligand_pos = None,
        org_protein_pos = None,
        ligand_bond_index = None, ligand_bond_type = None, ligand_bond_type_batch = None,
        protein_max_atom_num = None, ligand_max_atom_num  = None,
        args = None,
        protein_element = None,
        ligand_element  = None,
        sample = True,
        scale = True,
        ligand_atom_isring  = None,
        ligand_atom_isO     = None,
        ligand_atom_isN     = None,

        protein_atom_isring = None,
        protein_atom_isO    = None,
        protein_atom_isN    = None,

        cross_lig_isring_flag   = None,
        cross_lig_isO_flag      = None,
        cross_lig_isN_flag      = None,

        cross_pro_isring_flag   = None,
        cross_pro_isO_flag      = None,
        cross_pro_isN_flag      = None,


        cross_ligand    = None,
        cross_protein   = None,
        cross_distance  = None,

        
        cross_bond_index = None, 
        cross_bond_type = None, 
        cross_bond_index_reverse = None, 
        cross_bond_type_reverse = None,

        protein_coords_predict = None,

        complex_mol = None,

        protein_element_batch = None,
        protein_link_t_batch = None,
        protein_link_t_reverse_batch = None,

        ligand_element_batch = None,

        rd_pos = None,


        ) -> Tensor:


        #with torch.no_grad():
        with torch.enable_grad():
            outputs = model_forward_wrapper(
                model,
                None,
                None,
                xyzs, #
                None,
                sigma, #
                self.sigma_data,
                self.sigma_min,

                protein_pos=protein_pos, #
                protein_v=protein_v, 
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos, # #
                init_ligand_v=init_ligand_v,  
                batch_ligand=batch_ligand,
                time_step=time_step, #
                

                org_ligand_pos = org_ligand_pos,
                org_protein_pos = org_protein_pos,
                ligand_bond_index = ligand_bond_index, ligand_bond_type = ligand_bond_type, ligand_bond_type_batch = ligand_bond_type_batch,
                protein_max_atom_num = protein_max_atom_num, ligand_max_atom_num  = protein_max_atom_num,

                protein_element = protein_element,
                ligand_element  = ligand_element,
                sample = sample,
                scale = scale,

                ligand_atom_isring  = ligand_atom_isring,
                ligand_atom_isO     = ligand_atom_isO,
                ligand_atom_isN     = ligand_atom_isN,

                protein_atom_isring = protein_atom_isring,
                protein_atom_isO    = protein_atom_isO,
                protein_atom_isN    = protein_atom_isN,

                cross_lig_isring_flag   = cross_lig_isring_flag,
                cross_lig_isO_flag      = cross_lig_isO_flag,
                cross_lig_isN_flag      = cross_lig_isN_flag,

                cross_pro_isring_flag   = cross_pro_isring_flag,
                cross_pro_isO_flag      = cross_pro_isO_flag,
                cross_pro_isN_flag      = cross_pro_isN_flag,

                cross_ligand    = cross_ligand,
                cross_protein   = cross_protein,
                cross_distance  = cross_distance,

                cross_bond_index = cross_bond_index, 
                cross_bond_type = cross_bond_type, 
                cross_bond_index_reverse = cross_bond_index_reverse, 
                cross_bond_type_reverse = cross_bond_type_reverse,

                protein_coords_predict = protein_coords_predict,

                complex_mol = complex_mol,

                protein_element_batch = protein_element_batch,
                protein_link_t_batch = protein_link_t_batch,
                protein_link_t_reverse_batch = protein_link_t_reverse_batch,
                
                ligand_element_batch = ligand_element_batch,

                rd_pos = rd_pos,


                )

            '''
            outputs = {
                'pred_ligand_pos': final_ligand_pos,
                'pred_ligand_v': final_ligand_v,
                'final_pos': final_pos,
                'final_h': final_h, #
                'final_ligand_h': final_ligand_h,
                'atom_affinity': atom_affinity, #，，
                'final_exp_pred': final_exp_pred, #，，
                'batch_all': batch_all, #，mask_ligand
                'mask_ligand': mask_ligand,
                'ligand_v': ligand_v,
                'ligand_pos': init_ligand_pos,
                }
            '''

        
            batch_all, mask_ligand           = outputs['batch_all'], outputs['mask_ligand']
            atom_affinity, pred_affinity     = outputs['atom_affinity'], outputs['final_exp_pred']
            final_ligand_pos, final_ligand_h = outputs['pred_ligand_pos'], outputs['final_ligand_h']
            final_h     = outputs['final_h']
            ligand_v    = outputs['ligand_v']
            ligand_pos  = outputs['ligand_pos']
            final_ligand_v = outputs['pred_ligand_v']

            # pred_affinity = scatter_mean(self.expert_pred(final_h).squeeze(-1), batch_all)
            pred_affinity_log = pred_affinity.log()
            
            #，，consitency
            #type_grad = torch.autograd.grad(pred_affinity, ligand_v,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            type_grad = 0.0
            #pos_grad = torch.autograd.grad(pred_affinity_log, ligand_pos,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            pos_grad = 0.0

        

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'atom_affinity': atom_affinity,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h,
            'final_exp_pred': pred_affinity,
            'batch_all': batch_all,
            'mask_ligand': mask_ligand,
            'final_pos': outputs['final_pos']
        }
        return preds, type_grad, pos_grad
    

    def interpolate(
        self,
        model: nn.Module,
        a: Tensor,
        b: Tensor,
        ab_ratio: float,
        sigmas: Iterable[Union[Tensor, float]],
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Runs the interpolation  loop.

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        a : Tensor
            First reference sample.
        b : Tensor
            Second refernce sample.
        ab_ratio : float
            Ratio of the first reference sample to the second reference sample.
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1] range.
        verbose : bool, default=False
            Whether to display the progress bar.
        **kwargs : Any
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        Tensor
            Intepolated sample.
        """
        # Obtain latent samples from the initial samples
        a = a + sigmas[0] * torch.randn_like(a)
        b = b + sigmas[0] * torch.randn_like(b)

        # Perform spherical linear interpolation of the latents
        omega = torch.arccos(torch.sum((a / a.norm(p=2)) * (b / b.norm(p=2))))
        a = torch.sin(ab_ratio * omega) / torch.sin(omega) * a
        b = torch.sin((1 - ab_ratio) * omega) / torch.sin(omega) * b
        ab = a + b

        # Denoise the interpolated latents
        return self(
            model,
            ab,
            sigmas,
            start_from_y=True,
            add_initial_noise=False,
            clip_denoised=clip_denoised,
            verbose=verbose,
            **kwargs,
        )

    def __mask_transform(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> Tensor:
        return inverse_transform_fn(transform_fn(y) * (1.0 - mask) + x * mask)
