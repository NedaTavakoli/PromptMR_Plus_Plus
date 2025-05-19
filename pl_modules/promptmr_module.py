import torch
from data import transforms
from pl_modules import MriModule
from typing import List
import copy 
from mri_utils import SSIMLoss
import torch.nn.functional as F
import importlib

# Import the necessary functions for the patched forward method
from mri_utils.coil_combine import sens_expand, sens_reduce

def get_model_class(module_name, class_name="PromptMR"):
    """
    Dynamically imports the specified module and retrieves the class.

    Args:
        module_name (str): The module to import (e.g., 'model.m1', 'model.m2').
        class_name (str): The class to retrieve from the module (default: 'PromptMR').

    Returns:
        type: The imported class.
    """
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class


class PromptMrModule(MriModule):

    def __init__(
        self,
        num_cascades: int = 12,
        num_adj_slices: int = 5,
        n_feat0: int = 48,
        feature_dim: List[int] = [72,96,120],
        prompt_dim: List[int] = [24,48,72],
        sens_n_feat0: int = 24,
        sens_feature_dim: List[int] = [36,48,60],
        sens_prompt_dim: List[int] = [12,24,36],
        len_prompt: List[int] = [5,5,5],
        prompt_size: List[int] = [64,32,16],
        n_enc_cab: List[int] = [2,3,3],
        n_dec_cab: List[int] = [2,2,3],
        n_skip_cab: List[int] = [1,1,1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        learnable_prompt: bool = False,
        adaptive_input: bool = True,
        n_buffer: int = 4,
        n_history: int = 0,
        use_sens_adj: bool = True,
        model_version: str = "promptmr_v2",
        lr: float = 0.0002,
        lr_step_size: int = 11,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.01,
        use_checkpoint: bool = False,
        compute_sens_per_coil: bool = False,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational network.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            sens_n_feat0: Number of top-level feature channels for sense map
                estimation PromptUnet in PromptMR.
            sens_feature_dim: feature dim for each level in PromptUnet for
                sensitivity map estimation (SME) network.
            sens_prompt_dim: prompt dim for each level in PromptUnet in
                sensitivity map estimation (SME) network.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in BottleneckBlock.
            no_use_ca: not using channel attention.
            learnable_prompt: whether to set the prompt as learnable parameters.
            adaptive_input: whether to use adaptive input.
            n_buffer: number of buffer in adaptive input.
            n_history: number of historical feature aggregation, should be less than num_cascades.
            use_sens_adj: whether to use adjacent sensitivity map estimation.
            model_version: model version. Default is "promptmr_v2".
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            use_checkpoint: Whether to use checkpointing to trade compute for GPU memory.
            compute_sens_per_coil: (bool) whether to compute sensitivity maps per coil for memory saving
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices

        self.n_feat0 = n_feat0
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim

        self.sens_n_feat0 = sens_n_feat0
        self.sens_feature_dim = sens_feature_dim
        self.sens_prompt_dim = sens_prompt_dim

        self.len_prompt = len_prompt
        self.prompt_size = prompt_size
        self.n_enc_cab = n_enc_cab
        self.n_dec_cab = n_dec_cab
        self.n_skip_cab = n_skip_cab
        self.n_bottleneck_cab = n_bottleneck_cab

        self.no_use_ca = no_use_ca

        self.learnable_prompt = learnable_prompt
        self.adaptive_input = adaptive_input
        self.n_buffer = n_buffer
        self.n_history = n_history
        self.use_sens_adj = use_sens_adj
        # two flags for reducing memory usage
        self.use_checkpoint = use_checkpoint
        self.compute_sens_per_coil = compute_sens_per_coil
        
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.model_version = model_version
        PromptMR = get_model_class(f"models.{model_version}")  # Dynamically get the model class
        
        self.promptmr = PromptMR(
            num_cascades=self.num_cascades,
            num_adj_slices=self.num_adj_slices,
            n_feat0=self.n_feat0,
            feature_dim=self.feature_dim,
            prompt_dim=self.prompt_dim,
            sens_n_feat0=self.sens_n_feat0,
            sens_feature_dim=self.sens_feature_dim,
            sens_prompt_dim=self.sens_prompt_dim,
            len_prompt=self.len_prompt,
            prompt_size=self.prompt_size,
            n_enc_cab=self.n_enc_cab,
            n_dec_cab=self.n_dec_cab,
            n_skip_cab=self.n_skip_cab,
            n_bottleneck_cab=self.n_bottleneck_cab,
            no_use_ca=self.no_use_ca,
            learnable_prompt=learnable_prompt,
            n_history=self.n_history,
            n_buffer=self.n_buffer,
            adaptive_input=self.adaptive_input,
            use_sens_adj=self.use_sens_adj,
        )

        self.loss = SSIMLoss()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # step lr scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )
        return [optim], [scheduler]
    
    def forward(self, masked_kspace, mask, num_low_frequencies, mask_type="cartesian", use_checkpoint=False, compute_sens_per_coil=False):
        return self.promptmr(masked_kspace, mask, num_low_frequencies, mask_type, use_checkpoint=use_checkpoint, compute_sens_per_coil=compute_sens_per_coil)   

    def training_step(self, batch, batch_idx):
        # Get kspace data (either 'masked_kspace' or 'kspace')
        kspace_key = 'masked_kspace' if 'masked_kspace' in batch else 'kspace'
        
        # Get other parameters with defaults
        num_low_frequencies = batch.get('num_low_frequencies', None)
        mask_type = batch.get('mask_type', 'cartesian')
        
        # Forward pass
        output_dict = self(
            batch[kspace_key], 
            batch['mask'], 
            num_low_frequencies, 
            mask_type,
            use_checkpoint=self.use_checkpoint,
            compute_sens_per_coil=self.compute_sens_per_coil
        )
        
        output = output_dict['img_pred']
        
        # Check if target exists
        if 'target' in batch:
            # Center crop output and target to match
            target, output = transforms.center_crop_to_smallest(
                batch['target'], output)
            
            # Compute training loss
            max_value = batch.get('max_value', 1.0)
            loss = self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            )
            
            self.log("train_loss", loss, prog_bar=True)
            
            # Check for NaN loss
            if torch.isnan(loss):
                fname = batch.get('fname', 'unknown')
                slice_num = batch.get('slice_num', 'unknown')
                raise ValueError(f'nan loss on {fname} of slice {slice_num}')
            
            return loss
        else:
            # If no target is available, return zero loss
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=output.device)
            self.log("train_loss", dummy_loss, prog_bar=True)
            return dummy_loss

    def validation_step(self, batch, batch_idx):
        # Get kspace data (either 'masked_kspace' or 'kspace')
        kspace_key = 'masked_kspace' if 'masked_kspace' in batch else 'kspace'
        
        # Get other parameters with defaults
        num_low_frequencies = batch.get('num_low_frequencies', None)
        mask_type = batch.get('mask_type', 'cartesian')
        
        # Forward pass
        output_dict = self(
            batch[kspace_key], 
            batch['mask'], 
            num_low_frequencies, 
            mask_type,
            compute_sens_per_coil=self.compute_sens_per_coil
        )
        
        output = output_dict['img_pred']
        img_zf = output_dict['img_zf']
        
        # Check if target exists
        if 'target' in batch:
            # Center crop output and target to match
            target, output = transforms.center_crop_to_smallest(
                batch['target'], output)
            _, img_zf = transforms.center_crop_to_smallest(
                batch['target'], img_zf)
            
            # Compute validation loss
            max_value = batch.get('max_value', 1.0)
            val_loss = self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            )
        else:
            # If no target is available, skip loss computation
            target = None
            val_loss = torch.tensor(0.0, device=output.device)
        
        # Generate return dictionary
        result = {
            "batch_idx": batch_idx,
            "fname": batch.get('fname', 'unknown'),
            "slice_num": batch.get('slice_num', 0),
            "max_value": batch.get('max_value', 1.0),
            "img_zf": img_zf,
            "output": output,
            "loss": val_loss,
        }
        
        # Add other fields if available
        if kspace_key in batch:
            # For mask visualization (if available)
            if batch[kspace_key].shape[1] > 0:
                cc = batch[kspace_key].shape[1]
                centered_coil_visual = torch.log(1e-10 + torch.view_as_complex(batch[kspace_key][:, cc//2 if cc > 1 else 0]).abs())
                result["mask"] = centered_coil_visual
        
        # Add sensitivity maps if available
        if 'sens_maps' in output_dict:
            result["sens_maps"] = output_dict['sens_maps'][:, 0].abs() if output_dict['sens_maps'].dim() > 3 else output_dict['sens_maps'].abs()
        
        # Add target if available
        if target is not None:
            result["target"] = target
        
        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Handle dictionary batch input instead of object attributes
        kspace_key = 'masked_kspace' if 'masked_kspace' in batch else 'kspace'
        num_low_frequencies = batch.get('num_low_frequencies', None)
        
        # Default to 'cartesian' if mask_type is not present or is None
        mask_type = batch.get('mask_type', 'cartesian')
        if mask_type is None:
            mask_type = 'cartesian'
        
        output_dict = self(
            batch[kspace_key], 
            batch['mask'], 
            num_low_frequencies, 
            mask_type,
            compute_sens_per_coil=self.compute_sens_per_coil
        )
        
        output = output_dict['img_pred']

        # Handle crop size from batch dictionary
        crop_size = batch.get('crop_size', None)
        if crop_size is not None:
            if isinstance(crop_size[0], (list, tuple)):
                crop_size = [crop_size[0][0], crop_size[1][0]]  # if batch_size>1
            
            # Detect if output is smaller than crop size
            if output.shape[-1] < crop_size[1]:
                crop_size = (output.shape[-1], output.shape[-1])
                
            output = transforms.center_crop(output, crop_size)

        # Get number of slices if available
        num_slc = batch.get('num_slc', None)
        
        return {
            'output': output.cpu(), 
            'slice_num': batch.get('slice_num', 0), 
            'fname': batch.get('fname', 'unknown'),
            'num_slc': num_slc
        }

    def ensure_complex_dim(self, data):
        """
        Ensure the tensor has a complex dimension at the end.
        
        Args:
            data: Input tensor. If it doesn't have a complex dimension, we'll assume it's real only.
            
        Returns:
            Tensor with a complex dimension at the end.
        """
        if len(data.shape) > 0 and data.shape[-1] == 2:
            # Already has complex dimension
            return data
        
        # Add complex dimension (real part = data, imaginary part = zeros)
        zeros = torch.zeros_like(data)
        return torch.stack([data, zeros], dim=-1)

    # Update the forward method to use this function
    def forward(self, masked_kspace, mask, num_low_frequencies, mask_type="cartesian", use_checkpoint=False, compute_sens_per_coil=False):
        """
        Forward pass for PromptMR+ model.
        
        Args:
            masked_kspace: Masked k-space data
            mask: Sampling mask
            num_low_frequencies: Number of low frequency lines for center extraction
            mask_type: Type of mask ('cartesian' or 'poisson_disc')
            use_checkpoint: Whether to use checkpoint to save memory
            compute_sens_per_coil: Whether to compute sensitivity maps per coil to save memory
            
        Returns:
            Model output dictionary
        """
        # Import the necessary functions for the patched forward method
        from mri_utils.coil_combine import sens_expand, sens_reduce
        
        # Print original shapes for debugging
        print(f"[DEBUG] Original shapes - masked_kspace: {masked_kspace.shape}, mask: {mask.shape}")
        
        # Check if the tensor has 6 dimensions (one extra dimension)
        if len(masked_kspace.shape) == 6:
            print(f"[DEBUG] Detected 6D tensor with shape {masked_kspace.shape}")
            # Reshape to 5D by taking the first component of the 5th dimension
            masked_kspace = masked_kspace[..., 0, :]
            print(f"[DEBUG] Reshaped to 5D tensor with shape {masked_kspace.shape}")
        
        # Ensure both are float32
        masked_kspace = masked_kspace.to(torch.float32)
        
        # Required number of adjacent slices (hard-coded in the model)
        num_adj_slices = 5
        
        # Fix for einops rearrangement error with small second dimension
        if len(masked_kspace.shape) >= 2 and masked_kspace.shape[1] < num_adj_slices:
            print(f"[DEBUG] Second dimension {masked_kspace.shape[1]} is smaller than num_adj_slices ({num_adj_slices})")
            
            # Create a new tensor with the right dimensions for num_adj_slices
            expanded_kspace = torch.zeros(
                (masked_kspace.shape[0], num_adj_slices, *masked_kspace.shape[2:]),
                dtype=masked_kspace.dtype,
                device=masked_kspace.device
            )
            
            # Copy the data to the expanded tensor
            expanded_kspace[:, :masked_kspace.shape[1]] = masked_kspace
            masked_kspace = expanded_kspace
            print(f"[DEBUG] Expanded masked_kspace to shape {masked_kspace.shape}")
        
        # First, let's monkey-patch the PromptMRBlock.forward method to avoid using torch.where with mask
        original_forward = self.promptmr.cascades[0].forward
        
        def patched_forward(self_block, current_img, img_zf, latent, mask, sens_maps, history_feat=None):
            # This is a modified version of the forward method that doesn't use torch.where with mask
            current_kspace = sens_expand(current_img, sens_maps, self_block.num_adj_slices)
            # Skip using mask in torch.where, just use current_kspace directly
            ffx = sens_reduce(current_kspace, sens_maps, self_block.num_adj_slices)
            if self_block.model.n_buffer > 0:
                # adaptive input. buffer: A^H*A*x_i, s_i, x0, A^H*A*x_i-x0
                buffer = torch.cat([ffx, latent, img_zf] + [ffx-img_zf]*(self_block.model.n_buffer-3), dim=1)
            else:
                buffer = None
                
            soft_dc = (ffx - img_zf) * self_block.dc_weight
            model_term, latent, history_feat = self_block.model(current_img, history_feat, buffer)
            img_pred = current_img - soft_dc - model_term
            return img_pred, latent, history_feat
        
        # Patch all cascades' forward methods
        for cascade in self.promptmr.cascades:
            cascade.forward = patched_forward.__get__(cascade, type(cascade))
        
        # Now create a dummy mask (it won't be used for torch.where)
        # But it still needs to have the right shape for other operations
        dummy_mask = torch.ones(
            (masked_kspace.shape[0], masked_kspace.shape[2], masked_kspace.shape[3]),
            dtype=torch.bool,
            device=masked_kspace.device
        )
        
        # Call the model with our patched method
        try:
            output = self.promptmr(masked_kspace, dummy_mask, num_low_frequencies, mask_type, 
                            use_checkpoint=use_checkpoint, compute_sens_per_coil=compute_sens_per_coil)
        finally:
            # Restore original forward methods
            for cascade in self.promptmr.cascades:
                cascade.forward = original_forward.__get__(cascade, type(cascade))
        
        return output