import math
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm
from ...utils.utils_torch import kabsch_algorithm
from ...comparm import FGP
from ..min import opt_coords_moves,opt_complex_coords_moves
from typing import Iterator

from torch import Tensor, nn


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


def _update_ema_weights(
    ema_weight_iter: Iterator[Tensor],
    online_weight_iter: Iterator[Tensor],
    ema_decay_rate: float,
) -> None:
    for ema_weight, online_weight in zip(ema_weight_iter, online_weight_iter):
        #print (ema_weight.data,online_weight.data)
        if ema_weight.data is None or ema_weight.data.dtype==torch.long:
            ema_weight.data.copy_(online_weight.data)
        else:
            ema_weight.data.lerp_(online_weight.data, 1.0 - ema_decay_rate)


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
    # Update buffers
    _update_ema_weights(ema_model.buffers(), online_model.buffers(), ema_decay_rate)

    return ema_model
    
def timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 150,
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

    return num_timesteps + 1


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
    feats: Tensor,
    adjs: Tensor,
    xyzs: Tensor,
    gmasks: Tensor,
    sigma: Tensor,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
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
    c_skip = pad_dims_like(c_skip, xyzs).to(xyzs.device)
    c_out = pad_dims_like(c_out, xyzs).to(xyzs.device)

    sigma=sigma.to(xyzs.device)
    #if FGP.conf_model_version=='v1':
    #    feats,coords=model(feats=feats,coors=xyzs,mask=gmasks.bool(), edges=adjs, sigmas=sigma)
    #else:
    coords=model(feats=feats,coors=xyzs,mask=gmasks.bool(), edges=adjs, sigmas=sigma)
    mix_out=c_skip * xyzs + c_out *coords
    return mix_out

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
        initial_timesteps: int = 2,
        final_timesteps: int = 150,
    ) -> None:
    
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps

    def __call__(
        self,
        online_model: nn.Module,
        ema_model: nn.Module,
        feats: Tensor,
        adjs: Tensor,
        xyzs: Tensor,
        gmasks: Tensor,
        current_training_step: int,
        total_training_steps: int,
        fix_masks: Tensor=None,
        flex_masks: Tensor=None,
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
        fixlabels : Tensor
            fixlabels 
        **kwargs : Any
            Additional keyword arguments to be passed to the models.

        Returns
        -------
        (Tensor, Tensor)
            The predicted and target values for computing the loss.
        """
        num_timesteps = timesteps_schedule(
            current_training_step,
            total_training_steps,
            self.initial_timesteps,
            self.final_timesteps,
        )
        #print ('*'*80)
        #print (num_timesteps)
        sigmas = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, xyzs.device
        )
        #print (sigmas,sigmas.shape)
        noise = torch.randn_like(xyzs)
        #print (xyzs.shape[0])
        timesteps = torch.randint(0, num_timesteps - 1, (xyzs.shape[0],), device=xyzs.device)
        #print (noise.shape,timesteps.shape)
        current_sigmas = sigmas[timesteps]
        next_sigmas = sigmas[timesteps + 1]

        #print ('xyzs',xyzs)
        next_xyzs = xyzs + pad_dims_like(next_sigmas, xyzs) * noise
        if FGP.ec_mode!='inpaint':
        #print (next_xyzs.shape)
            assert (flex_masks is not None) and (fix_masks is not None), "flex_masks and fix_masks should be provided if ecmode != inpaint !"
            #print (flex_masks.shape,fix_masks.shape)
            next_xyzs=next_xyzs*flex_masks.unsqueeze(-1)+xyzs*fix_masks.unsqueeze(-1)
        
        next_xyzs = model_forward_wrapper(
            online_model,
            feats,
            adjs,
            next_xyzs,
            gmasks,
            next_sigmas,
            self.sigma_data,
            self.sigma_min,
        )
        
        #print ('predicted_next_xyzs',next_xyzs)
        with torch.no_grad():
            current_xyzs = xyzs + pad_dims_like(current_sigmas, xyzs) * noise
            if FGP.ec_mode!='inpaint':
                #print (next_xyzs.shape)
                assert (flex_masks is not None) and (fix_masks is not None), "flex_masks and fix_masks should be provided if ecmode != inpaint !"
                current_xyzs=current_xyzs*flex_masks.unsqueeze(-1)+xyzs*fix_masks.unsqueeze(-1)
            current_xyzs = model_forward_wrapper(
                ema_model,
                feats,
                adjs,
                current_xyzs,
                gmasks,
                current_sigmas,
                self.sigma_data,
                self.sigma_min,
            )
        
        return (next_xyzs, current_xyzs)

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

    def __call__(
        self,
        model: nn.Module,
        atoms:Tensor,
        feats: Tensor,
        adjs: Tensor,
        ligand_labels:Tensor,
        pocket_labels:Tensor,
        y: Tensor,
        gmasks: Tensor,
        sigmas: Iterable[Union[Tensor, float]],

        mask: Optional[Tensor] = None,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        start_from_y: bool = False,
        add_initial_noise: bool = True,
        clip_denoised: bool = False,
        verbose: bool = False,
        with_MMFF_guide: bool = False,
        guide_loops=1,
        guide_type='asynchronous',
        show_state=False,
        min_type='LBFGS',
        rdkit_mols=None,
        opt_types="complex",
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
        # Set mask to all ones which is useful for sampling and style transfer
        if mask is None:
            mask = torch.ones_like(y)
            fix_mask=((1-mask)*gmasks.unsqueeze(-1)).bool()
            flex_mask=mask*gmasks.unsqueeze(-1).bool()
        else:
            #print (mask.shape,gmasks.shape)
            fix_mask=torch.tile(((1-mask)*gmasks.unsqueeze(-1)).bool(),(1,1,3))
            flex_mask=torch.tile((mask*gmasks.unsqueeze(-1)).bool(),(1,1,3))
        
        b=fix_mask.shape[0]
        d=fix_mask.shape[-1]
        
        #print ('fix_mask',fix_mask.shape)
        fix_part_coords=y[fix_mask].reshape((b,-1,d))
        #print (fix_part_coords[0])
        x = y if start_from_y else torch.zeros_like(y)

        # Sample at the end of the schedule
        x_list=[]
        x_list_before_transform=[]
        
        y = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)
        
        ligand_atoms=atoms*ligand_labels
        pocket_atoms=atoms*pocket_labels
        
        # For tasks like interpolation where noise will already be added in advance we
        # can skip the noising process        
        x = y + sigmas[0] * torch.randn_like(y) if add_initial_noise else y
        if FGP.ec_mode!='inpaint':
            x=x*flex_mask+y*fix_mask

        x_list.append(x.unsqueeze(1)) 
        x_list_before_transform.append(x.unsqueeze(1))
        
        sigma = torch.full((x.shape[0],), sigmas[0], dtype=x.dtype, device=x.device)
        x_bp=x.clone().detach()
        x = model_forward_wrapper(model, feats, adjs, x, gmasks, sigma, self.sigma_data, self.sigma_min)

        if True:
        #if FGP.ec_mode=='inpaint': 
            x_fix_part=x[fix_mask].reshape((b,-1,d))
            T,R=kabsch_algorithm(x_fix_part,fix_part_coords)
            x=x@R-T
            ligand_x=x*ligand_labels.unsqueeze(-1)
            #print (ligand_x.shape,ligand_atoms.shape,torch.sum(ligand_x*ligand_atoms.unsqueeze(-1),axis=-2).shape,torch.sum(ligand_atoms,axis=-1,keepdim=True).shape)
            ligand_mass_center=torch.sum(ligand_x*ligand_atoms.unsqueeze(-1),axis=-2)/torch.sum(ligand_atoms,axis=-1,keepdim=True)
            if torch.sum(pocket_labels)>0:
                pocket_x=x*pocket_labels.unsqueeze(-1)
                pocket_mass_center=torch.sum(pocket_x*pocket_atoms.unsqueeze(-1),axis=-2)/torch.sum(pocket_atoms,axis=-1,keepdim=True)
                #x=x-ligand_mass_center.unsqueeze(1)
                x=x-pocket_mass_center.unsqueeze(1)
        
        if with_MMFF_guide:
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
            if guide_type=='synchronous':
                if opt_types=="complex":
                    x_moves,energy_min=opt_complex_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask,pocket_masks=pocket_labels.bool(),ligand_masks=ligand_labels.bool())
                else:
                    x_moves,energy_min=opt_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask)
            
            #，，。+complex
            else:
                if opt_types=="complex":
                    x_moves,energy_min=opt_complex_coords_moves(x,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask,pocket_masks=pocket_labels.bool(),ligand_masks=ligand_labels.bool())
                else:
                    x_moves,energy_min=opt_coords_moves(x,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask)

            x=x+x_moves*flex_mask #x_moves，，+

        x_list_before_transform.append(x.unsqueeze(1))

         
        x = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)
        x_list.append(x.unsqueeze(1))
        
        # Progressively denoise the sample and skip the first step as it has already
        # been run
        
        pbar = tqdm(sigmas[1:], disable=(not verbose))
        
        for sigma in pbar:
            pbar.set_description(f"sampling (σ={sigma:.4f})")
            sigma = torch.full((x.shape[0],), sigma, dtype=x.dtype, device=x.device)

            x = x + pad_dims_like(
                (sigma**2 - self.sigma_min**2) ** 0.5, x
            ) * torch.randn_like(x)
            if FGP.ec_mode!='inpaint':
                x=x*flex_mask+y*fix_mask
            x_bp=x.clone().detach()
            x = model_forward_wrapper(
                    model, feats, adjs, x, gmasks, sigma, self.sigma_data, self.sigma_min
                )
            
            if True:
                x_fix_part=x[fix_mask].reshape((b,-1,d))
                T,R=kabsch_algorithm(x_fix_part,fix_part_coords)
                x=x@R-T
                ligand_x=x*ligand_labels.unsqueeze(-1)
                ligand_mass_center=torch.sum(ligand_x*ligand_atoms.unsqueeze(-1),axis=-2)/torch.sum(ligand_atoms,axis=1,keepdim=True)
                
                if torch.sum(pocket_labels)>0:
                    pocket_x=x*pocket_labels.unsqueeze(-1)
                    pocket_mass_center=torch.sum(pocket_x*pocket_atoms.unsqueeze(-1),axis=-2)/torch.sum(pocket_atoms,axis=-1,keepdim=True)    
                    #x=x-ligand_mass_center.unsqueeze(1)
                    x=x-pocket_mass_center.unsqueeze(1)

            if with_MMFF_guide:
                if guide_type=='synchronous':
                    if opt_types=="complex":
                        x_moves,energy_min=opt_complex_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask,pocket_masks=pocket_labels.bool(),ligand_masks=ligand_labels.bool())
                    else:
                        x_moves,energy_min=opt_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask)
                else:
                    if opt_types=="complex":
                        x_moves,energy_min=opt_complex_coords_moves(x,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask,pocket_masks=pocket_labels.bool(),ligand_masks=ligand_labels.bool())
                    else:
                        x_moves,energy_min=opt_coords_moves(x,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask)
                
                x=x+x_moves*flex_mask

            x_list_before_transform.append(x.unsqueeze(1))
            
            x = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)
            x_list.append(x.unsqueeze(1))
        if with_MMFF_guide:
            return x,x_list,x_list_before_transform,energy_min
        else:
            return x,x_list,x_list_before_transform,None

    def __mask_transform(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> Tensor:
        return inverse_transform_fn(transform_fn(y) * (1.0 - mask) + x * mask)
