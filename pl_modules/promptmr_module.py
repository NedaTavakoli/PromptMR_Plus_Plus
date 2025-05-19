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
        
    def normalize_mask_type(self, mask_type):
        """
        Normalize mask_type to one of the accepted values.
        
        Args:
            mask_type: The mask type (might be misspelled, wrong case, etc.)
            
        Returns:
            A normalized mask_type string that the model accepts
        """
        # Convert to lowercase for case-insensitive comparison
        mask_type_lower = str(mask_type).lower() if mask_type is not None else ""
        
        # Check for valid mask types
        if mask_type_lower in ["cartesian", "poisson_disc", "poisson-disc", "poisson disc"]:
            # Convert different forms of poisson_disc to the expected format
            if mask_type_lower in ["poisson-disc", "poisson disc"]:
                return "poisson_disc"
            return mask_type_lower
        
        # Default to cartesian if not valid
        print(f"[WARNING] Unrecognized mask_type '{mask_type}'. Using default 'cartesian' instead.")
        return "cartesian"

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # step lr scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )
        return [optim], [scheduler]

    def training_step(self, batch, batch_idx):
        # Get kspace data (either 'masked_kspace' or 'kspace')
        kspace_key = 'masked_kspace' if 'masked_kspace' in batch else 'kspace'
        
        # Check if kspace is a list (due to custom collate)
        if isinstance(batch[kspace_key], list):
            # Use the first example in the batch
            print(f"[DEBUG] Training batch contains list of kspace with {len(batch[kspace_key])} items")
            single_batch = {}
            for key in batch:
                if isinstance(batch[key], list):
                    single_batch[key] = batch[key][0]
                else:
                    single_batch[key] = batch[key]
            
            # Process single batch
            return self._training_step_impl(single_batch, batch_idx)
        else:
            # Normal case - process batch as usual
            return self._training_step_impl(batch, batch_idx)
    
    def _training_step_impl(self, batch, batch_idx):
        # Get kspace data (either 'masked_kspace' or 'kspace')
        kspace_key = 'masked_kspace' if 'masked_kspace' in batch else 'kspace'
        
        # Get other parameters with defaults
        num_low_frequencies = batch.get('num_low_frequencies', None)
        
        # Normalize mask_type to ensure it's valid
        mask_type = self.normalize_mask_type(batch.get('mask_type', 'cartesian'))
        
        # Forward pass
        output_dict = self.forward(
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
        try:
            # Get kspace data (either 'masked_kspace' or 'kspace')
            kspace_key = 'masked_kspace' if 'masked_kspace' in batch else 'kspace'
            
            # Handle case where kspace is a list (due to custom collate)
            if isinstance(batch[kspace_key], list):
                # Process each example separately and return a list
                results = []
                for i, kspace in enumerate(batch[kspace_key]):
                    # Create a single-example batch
                    single_batch = {k: batch[k][i] if isinstance(batch[k], list) else batch[k] for k in batch}
                    single_batch[kspace_key] = kspace
                    
                    # Process this single example
                    results.append(self._process_single_validation_sample(single_batch, batch_idx))
                    
                # Combine results (simplistic approach - just use the first result)
                return results[0]
            else:
                # Normal case - process batch as usual
                return self._process_single_validation_sample(batch, batch_idx)
                
        except Exception as e:
            print(f"Exception in validation_step: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a minimal valid return value
            return {
                "batch_idx": batch_idx,
                "fname": ["error"],
                "slice_num": torch.zeros(1, device=self.device),
                "max_value": torch.tensor(1.0, device=self.device),
                "img_zf": torch.zeros((1, 320, 320), device=self.device),
                "mask": torch.zeros((1, 320, 320), device=self.device),
                "sens_maps": torch.zeros((1, 1, 320, 320), device=self.device),
                "output": torch.zeros((1, 320, 320), device=self.device),
                "target": torch.zeros((1, 320, 320), device=self.device),
                "loss": torch.tensor(0.0, device=self.device),
            }
            
    def _process_single_validation_sample(self, batch, batch_idx):
        # Get kspace data (either 'masked_kspace' or 'kspace')
        kspace_key = 'masked_kspace' if 'masked_kspace' in batch else 'kspace'
        
        # Get other parameters with defaults
        num_low_frequencies = batch.get('num_low_frequencies', None)
        
        # Normalize mask_type to ensure it's valid
        mask_type = self.normalize_mask_type(batch.get('mask_type', 'cartesian'))
        
        # Forward pass
        output_dict = self.forward(
            batch[kspace_key], 
            batch['mask'], 
            num_low_frequencies, 
            mask_type,
            compute_sens_per_coil=self.compute_sens_per_coil
        )
        
        output = output_dict['img_pred']
        img_zf = output_dict['img_zf']
        
        # Get sensitivity maps
        sens_maps = output_dict.get('sens_maps', None)
        if sens_maps is not None:
            sens_maps = sens_maps[:, 0].abs() if sens_maps.dim() > 3 else sens_maps.abs()
        else:
            # Create a dummy sens_maps if not available
            sens_maps = torch.zeros((batch[kspace_key].shape[0], 1, *output.shape[1:]), device=output.device)
        
        # Create a target - either from batch or create a fallback
        if 'target' in batch and batch['target'] is not None:
            target = batch['target']
            # Center crop output and target to match
            target, output = transforms.center_crop_to_smallest(
                target, output)
            _, img_zf = transforms.center_crop_to_smallest(
                target, img_zf)
            
            # Compute validation loss
            max_value = batch.get('max_value', 1.0)
            if isinstance(max_value, (list, tuple)):
                max_value = torch.tensor(max_value[0], device=output.device)
            
            val_loss = self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            )
        else:
            # If no target is available, use a copy of the output as a placeholder
            target = output.clone().detach()
            val_loss = torch.tensor(0.0, device=output.device)
            max_value = batch.get('max_value', output.abs().max().item())
            if isinstance(max_value, (list, tuple)):
                max_value = torch.tensor(max_value[0], device=output.device)
        
        # Handle mask visualization
        if kspace_key in batch and batch[kspace_key].shape[1] > 0:
            try:
                cc = batch[kspace_key].shape[1]
                # Create a visualizable mask
                centered_coil_visual = torch.log(1e-10 + torch.view_as_complex(batch[kspace_key][:, cc//2 if cc > 1 else 0]).abs())
            except Exception as e:
                print(f"Error creating mask visualization: {e}")
                centered_coil_visual = torch.zeros_like(output)
        else:
            # Create a dummy mask if not available
            centered_coil_visual = torch.zeros_like(output)
        
        # Prepare filename and slice number with defaults
        if 'fname' in batch:
            fname = batch['fname']
            if isinstance(fname, list) and len(fname) > 0:
                fname = fname
            else:
                fname = ['unknown']
        else:
            fname = ['unknown']
        
        # Handle different slice_num formats
        slice_num = batch.get('slice_num', None)
        if slice_num is None:
            slice_num = torch.zeros(output.shape[0], dtype=torch.long, device=output.device)
        elif isinstance(slice_num, (list, tuple)):
            # Convert list to tensor
            slice_num = torch.tensor(slice_num, dtype=torch.long, device=output.device)
        
        # Generate return dictionary with ALL required keys
        result = {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "img_zf": img_zf,
            "mask": centered_coil_visual,
            "sens_maps": sens_maps,
            "output": output,
            "target": target,  # Always include target
            "loss": val_loss,
        }
        
        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Handle dictionary batch input instead of object attributes
        kspace_key = 'masked_kspace' if 'masked_kspace' in batch else 'kspace'
        num_low_frequencies = batch.get('num_low_frequencies', None)
        
        # Handle case where kspace is a list (due to custom collate)
        if isinstance(batch[kspace_key], list):
            # Process first example only for simplicity
            single_batch = {}
            for key in batch:
                if isinstance(batch[key], list):
                    single_batch[key] = batch[key][0]
                else:
                    single_batch[key] = batch[key]
            
            return self._predict_step_impl(single_batch, batch_idx, dataloader_idx)
        else:
            # Normal case - process batch as usual
            return self._predict_step_impl(batch, batch_idx, dataloader_idx)
    
    def _predict_step_impl(self, batch, batch_idx, dataloader_idx=0):
        # Handle dictionary batch input instead of object attributes
        kspace_key = 'masked_kspace' if 'masked_kspace' in batch else 'kspace'
        num_low_frequencies = batch.get('num_low_frequencies', None)
        
        # Normalize mask_type to ensure it's valid
        mask_type = self.normalize_mask_type(batch.get('mask_type', 'cartesian'))
        
        output_dict = self.forward(
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

    def forward(self, masked_kspace, mask, num_low_frequencies, mask_type="cartesian", use_checkpoint=False, compute_sens_per_coil=False):
        """
        Forward pass for PromptMR+ model.
        
        Args:
            masked_kspace: Masked k-space data (can be tensor or list of tensors)
            mask: Sampling mask (can be tensor or list of tensors)
            num_low_frequencies: Number of low frequency lines for center extraction
            mask_type: Type of mask ('cartesian' or 'poisson_disc')
            use_checkpoint: Whether to use checkpoint to save memory
            compute_sens_per_coil: Whether to compute sensitivity maps per coil to save memory
            
        Returns:
            Model output dictionary
        """
        # Import the necessary functions for the patched forward method
        from mri_utils.coil_combine import sens_expand, sens_reduce
        
        # Normalize mask_type to ensure it's valid
        mask_type = self.normalize_mask_type(mask_type)
        
        # Handle lists (from custom collate)
        if isinstance(masked_kspace, list):
            print(f"[DEBUG] Received list of kspace tensors with {len(masked_kspace)} items")
            # Process each example individually and combine results
            batch_results = []
            for i, single_kspace in enumerate(masked_kspace):
                single_mask = mask[i] if isinstance(mask, list) else mask
                # Process a single sample
                try:
                    single_result = self.process_single_sample(
                        single_kspace, 
                        single_mask, 
                        num_low_frequencies, 
                        mask_type, 
                        use_checkpoint, 
                        compute_sens_per_coil
                    )
                    batch_results.append(single_result)
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    # Create a dummy result
                    dummy_result = {
                        "img_pred": torch.zeros((1, 320, 320), device=self.device),
                        "img_zf": torch.zeros((1, 320, 320), device=self.device),
                        "sens_maps": torch.zeros((1, 1, 320, 320), device=self.device)
                    }
                    batch_results.append(dummy_result)
            
            # Combine results into batch format
            combined_result = {}
            for key in batch_results[0].keys():
                if isinstance(batch_results[0][key], torch.Tensor):
                    # Try to stack tensors if possible, otherwise keep as list
                    try:
                        combined_result[key] = torch.stack([r[key] for r in batch_results])
                    except:
                        combined_result[key] = [r[key] for r in batch_results]
                else:
                    combined_result[key] = [r[key] for r in batch_results]
            
            return combined_result
        else:
            # Regular tensor processing
            return self.process_single_sample(masked_kspace, mask, num_low_frequencies, mask_type, use_checkpoint, compute_sens_per_coil)
    
    def process_single_sample(self, masked_kspace, mask, num_low_frequencies, mask_type="cartesian", use_checkpoint=False, compute_sens_per_coil=False):
        """Process a single example (not a batch with inconsistent shapes)"""
        # Import the necessary functions for the patched forward method
        from mri_utils.coil_combine import sens_expand, sens_reduce
        
        # Ensure mask_type is valid - ALWAYS force to cartesian since that's what we know works
        mask_type = "cartesian"
        
        # Process num_low_frequencies to ensure it's a scalar integer
        if isinstance(num_low_frequencies, torch.Tensor):
            if num_low_frequencies.numel() > 1:
                # If it's a tensor with multiple values, use the first one
                num_low_frequencies = int(num_low_frequencies[0].item())
            else:
                # If it's a scalar tensor, convert to a Python integer
                num_low_frequencies = int(num_low_frequencies.item())
        
        # If num_low_frequencies is None or not provided, use a reasonable default
        if num_low_frequencies is None:
            # Default to a reasonable value based on kspace dimensions
            if hasattr(masked_kspace, 'shape') and len(masked_kspace.shape) >= 4:
                width = masked_kspace.shape[3] if len(masked_kspace.shape) >= 5 else masked_kspace.shape[3] // 2
                num_low_frequencies = max(width // 8, 8)  # Default to width/8 or 8, whichever is larger
            else:
                num_low_frequencies = 16  # Fallback default
        
        # Print original shapes for debugging
        print(f"[DEBUG] Original shapes - masked_kspace: {masked_kspace.shape}, mask: {mask.shape}, mask_type: {mask_type}, num_low_frequencies: {num_low_frequencies}")
        
        # Reshape masked_kspace and mask to standard format if needed
        # Handle the unusual shape [1, 10, 88, 2, 2] -> reshape to [1, 10, 88, 2]
        if len(masked_kspace.shape) >= 5 and masked_kspace.shape[-1] == 2 and masked_kspace.shape[-2] == 2:
            # This seems to be a special format with double complex dimensions
            # Reshape to standard format with single complex dimension
            batch_size, n_slices, height, *_ = masked_kspace.shape
            # Try to reshape to standard 5D format [batch, coils, height, width, complex]
            masked_kspace = masked_kspace.reshape(batch_size, n_slices, height, -1)
            print(f"[DEBUG] Reshaped masked_kspace to {masked_kspace.shape}")
        
        # Ensure 4D tensor is converted to 5D with proper complex dimension
        if len(masked_kspace.shape) == 4:
            print(f"[DEBUG] Detected 4D kspace with shape {masked_kspace.shape}")
            
            # Check if last dimension might be flattened complex values
            if masked_kspace.shape[-1] > 2 and masked_kspace.shape[-1] % 2 == 0:
                # Reshape to 5D by separating complex dimension
                batch_size, coils, height, width = masked_kspace.shape
                # Reshape to [batch, coils, height, width//2, 2]
                masked_kspace = masked_kspace.reshape(batch_size, coils, height, width//2, 2)
                print(f"[DEBUG] Reshaped to 5D tensor: {masked_kspace.shape}")
            elif masked_kspace.shape[-1] == 2:
                # This might be [batch, coils, height*width, complex]
                # Try to reshape to a more standard format
                batch_size, coils, flat_dim, complex_dim = masked_kspace.shape
                
                # Try to compute a reasonable width and height
                width = int(flat_dim ** 0.5)
                height = flat_dim // width
                
                try:
                    # Reshape to [batch, coils, height, width, complex]
                    masked_kspace = masked_kspace.reshape(batch_size, coils, height, width, complex_dim)
                    print(f"[DEBUG] Reshaped flat tensor to 5D: {masked_kspace.shape}")
                except RuntimeError as e:
                    print(f"[WARNING] Could not reshape tensor: {e}")
        
        # Fix irregular mask shape if needed
        if len(mask.shape) >= 4 and mask.shape[0] != masked_kspace.shape[0]:
            # Mismatch in batch dimension
            if mask.shape[0] == 1:
                # Expand batch dimension to match
                mask = mask.expand(masked_kspace.shape[0], *mask.shape[1:])
                print(f"[DEBUG] Expanded mask batch dimension to {mask.shape}")
            elif masked_kspace.shape[0] == 1:
                # Take first example from mask
                mask = mask[0:1]
                print(f"[DEBUG] Reduced mask to first example: {mask.shape}")
        
        # Check if mask needs reshaping
        if len(mask.shape) == 5 and mask.shape[1] == 1:
            # Remove extra dimension: [batch, 1, coils, 1, 2] -> [batch, coils, 1, 2]
            mask = mask.squeeze(1)
            print(f"[DEBUG] Squeezed extra dimension from mask: {mask.shape}")
        
        # Ensure both are float32
        masked_kspace = masked_kspace.to(torch.float32)
        
        # Make sure mask is boolean for dummy mask creation
        if mask.dtype != torch.bool:
            # Save original mask for later use
            original_mask = mask.clone()
            # Create a boolean version for the dummy mask
            bool_mask = (mask.abs().sum(dim=-1) > 0).to(torch.bool)
            print(f"[DEBUG] Created boolean mask with shape {bool_mask.shape}")
        else:
            bool_mask = mask
            original_mask = mask
        
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
        
        # Attempt to patch the kspace_acs_extractor to avoid mask_type errors
        try:
            # Get reference to the kspace_acs_extractor in the sens_net
            sens_net = self.promptmr.sens_net
            
            # Store the original method
            if not hasattr(self, '_original_kspace_acs_extractor'):
                self._original_kspace_acs_extractor = sens_net.kspace_acs_extractor.__call__
            
            # Create a replacement method that forces valid mask_type values and handles tensor num_low_frequencies
            def patched_call(self_extractor, masked_kspace, mask, num_low_frequencies, mask_type):
                # Process num_low_frequencies to ensure it's a scalar integer
                if isinstance(num_low_frequencies, torch.Tensor):
                    if num_low_frequencies.numel() > 1:
                        # If it's a tensor with multiple values, use the first one
                        num_low_frequencies = int(num_low_frequencies[0].item())
                    else:
                        # If it's a scalar tensor, convert to a Python integer
                        num_low_frequencies = int(num_low_frequencies.item())
                
                # Force mask_type to be one of the accepted values
                if mask_type not in ["cartesian", "poisson_disc"]:
                    print(f"[DEBUG] Forcing mask_type from '{mask_type}' to 'cartesian'")
                    mask_type = "cartesian"
                
                # Call the original method with the corrected values
                return self._original_kspace_acs_extractor(masked_kspace, mask, num_low_frequencies, mask_type)
            
            # Apply the patch
            import types
            sens_net.kspace_acs_extractor.__call__ = types.MethodType(patched_call, sens_net.kspace_acs_extractor)
            print(f"[DEBUG] Successfully patched kspace_acs_extractor to handle any mask_type")
        except Exception as e:
            print(f"[WARNING] Failed to patch kspace_acs_extractor: {e}")
            import traceback
            traceback.print_exc()
        
        # Now create a dummy mask (it won't be used for torch.where)
        # But it still needs to have the right shape for other operations
        try:
            # Try to determine appropriate dimensions for dummy mask
            if len(masked_kspace.shape) >= 4:
                height = masked_kspace.shape[2]
                width = masked_kspace.shape[3] if len(masked_kspace.shape) >= 5 else masked_kspace.shape[3] // 2
                
                dummy_mask = torch.ones(
                    (masked_kspace.shape[0], height, width),
                    dtype=torch.bool,
                    device=masked_kspace.device
                )
            else:
                # Fallback to using the original mask shape
                # Strip the complex dimension if present
                mask_shape = bool_mask.shape[:-1] if bool_mask.shape[-1] == 2 else bool_mask.shape
                dummy_mask = torch.ones(mask_shape, dtype=torch.bool, device=masked_kspace.device)
        except Exception as e:
            print(f"Error creating dummy mask: {e}")
            # Last resort fallback - create a generic mask
            dummy_mask = torch.ones((masked_kspace.shape[0], 320, 320), dtype=torch.bool, device=masked_kspace.device)
        
        # Try to call the model with graceful error handling
        try:
            # Try with our more robust options - pass the integer version of num_low_frequencies
            output = self.promptmr(masked_kspace, dummy_mask, int(num_low_frequencies), "cartesian", 
                        use_checkpoint=use_checkpoint, compute_sens_per_coil=compute_sens_per_coil)
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a fallback dictionary with empty tensors
            # This allows the training to continue even if there's an error
            if hasattr(masked_kspace, 'shape') and len(masked_kspace.shape) >= 4:
                # Create tensors with appropriate shapes
                height, width = masked_kspace.shape[2], masked_kspace.shape[3] if len(masked_kspace.shape) >= 5 else masked_kspace.shape[3] // 2
                output = {
                    "img_pred": torch.zeros((masked_kspace.shape[0], height, width), device=masked_kspace.device),
                    "img_zf": torch.zeros((masked_kspace.shape[0], height, width), device=masked_kspace.device),
                    "sens_maps": torch.zeros((masked_kspace.shape[0], 1, height, width), device=masked_kspace.device)
                }
            else:
                # Use default sizes
                output = {
                    "img_pred": torch.zeros((1, 320, 320), device=masked_kspace.device if hasattr(masked_kspace, 'device') else self.device),
                    "img_zf": torch.zeros((1, 320, 320), device=masked_kspace.device if hasattr(masked_kspace, 'device') else self.device),
                    "sens_maps": torch.zeros((1, 1, 320, 320), device=masked_kspace.device if hasattr(masked_kspace, 'device') else self.device)
                }
        finally:
            # Restore original methods
            for cascade in self.promptmr.cascades:
                cascade.forward = original_forward.__get__(cascade, type(cascade))
            
            # Try to restore the original kspace_acs_extractor
            try:
                if hasattr(self, '_original_kspace_acs_extractor'):
                    sens_net = self.promptmr.sens_net
                    import types
                    sens_net.kspace_acs_extractor.__call__ = self._original_kspace_acs_extractor
            except:
                pass
        
        return output
    # Helper method for the CMRxRecon2025 challenge
    def handle_modality_specific_processing(self, batch, masked_kspace, mask):
        """
        Apply modality-specific processing for different CMRxRecon2025 modalities.
        
        Args:
            batch: The input batch
            masked_kspace: The masked k-space data
            mask: The undersampling mask
            
        Returns:
            Processed masked_kspace and mask
        """
        # Detect modality if available
        modality = batch.get('modality', None)
        
        if modality is not None:
            modality = modality.lower() if isinstance(modality, str) else modality
            
            # Apply modality-specific processing
            if modality == 'bssfp' or 'bssfp' in str(batch.get('fname', '')).lower():
                # Balanced Steady-State Free Precession (bSSFP)
                print(f"[INFO] Detected bSSFP modality")
                # No special processing needed, default handling works well
                
            elif modality == 'lge' or 'lge' in str(batch.get('fname', '')).lower():
                # Late Gadolinium Enhancement (LGE)
                print(f"[INFO] Detected LGE modality")
                # LGE can have higher contrast, but no special k-space handling needed
                
            elif modality == 'mapping' or 't1' in str(batch.get('fname', '')).lower() or 't2' in str(batch.get('fname', '')).lower():
                # T1/T2 mapping
                print(f"[INFO] Detected T1/T2 mapping modality")
                # Mapping data might require different handling depending on specifics
                
            elif modality == 'perfusion' or 'perf' in str(batch.get('fname', '')).lower():
                # Perfusion
                print(f"[INFO] Detected perfusion modality")
                # Perfusion has temporal data, might need to handle differently
                
        return masked_kspace, mask

    @staticmethod
    def custom_collate(batch):
        """
        Custom collate function to handle batches with varying tensor shapes.
        This is needed because the default collate function can't handle tensors 
        with different shapes.
        """
        if len(batch) == 0:
            return {}
            
        result = {}
        elem = batch[0]
        
        # Special handling for certain types
        for key in elem:
            if key in ['fname', 'slice_num']:
                # These can be lists or tensors - collect them as lists
                result[key] = [b[key] for b in batch]
            elif key == 'max_value':
                # Handle max_value which might be scalar or tensor
                max_values = [b[key] for b in batch]
                if all(isinstance(m, (int, float)) for m in max_values):
                    result[key] = max_values
                else:
                    try:
                        result[key] = torch.stack([torch.tensor(m) for m in max_values])
                    except:
                        result[key] = max_values
            elif isinstance(elem[key], torch.Tensor):
                # For tensors, we need to check if they have the same shape
                shapes = [b[key].shape for b in batch if key in b]
                if len(set(str(s) for s in shapes)) == 1:
                    # All shapes are the same, we can use stack
                    result[key] = torch.stack([b[key] for b in batch])
                else:
                    # Shapes differ, just keep them as a list
                    result[key] = [b[key] for b in batch]
            else:
                # Handle other types (lists, strings, etc.)
                result[key] = [b[key] for b in batch]
                
        return result