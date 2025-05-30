"""
Description: This script is the main entry point for the LightningCLI.
"""
import os
import sys
from itertools import chain
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
import yaml
import torch
import numpy as np
import types
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import BasePredictionWriter

# from mri_utils import save_reconstructions
from mri_utils.utils import save_reconstructions  # Import directly from the utils module

from pl_modules import PromptMrModule

def custom_collate(batch):
    """
    Custom collate function to handle batches with varying tensor shapes.
    """
    if len(batch) == 0:
        return {}
        
    result = {}
    elem = batch[0]
    
    # Handle batch where elements are dicts
    if isinstance(elem, dict):
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
                # For tensors, handle different shapes
                try:
                    if key == 'kspace' or key == 'masked_kspace':
                        # For kspace data, just collect as a list if shapes don't match
                        if all(b[key].shape[2:] == elem[key].shape[2:] for b in batch):
                            result[key] = default_collate([b[key] for b in batch])
                        else:
                            result[key] = [b[key] for b in batch]
                    elif key == 'mask':
                        # For mask data, just collect as a list if shapes don't match
                        if all(b[key].shape == elem[key].shape for b in batch):
                            result[key] = default_collate([b[key] for b in batch])
                        else:
                            result[key] = [b[key] for b in batch]
                    else:
                        # Try standard collate for other tensor keys
                        result[key] = default_collate([b[key] for b in batch])
                except:
                    # If default_collate fails, just collect as a list
                    result[key] = [b[key] for b in batch]
            else:
                # Handle other types (lists, strings, etc.)
                try:
                    result[key] = default_collate([b[key] for b in batch])
                except:
                    result[key] = [b[key] for b in batch]
    else:
        # Just use default_collate for non-dict batch elements
        try:
            return default_collate(batch)
        except:
            return batch
            
    return result

def preprocess_save_dir():
    """Ensure `save_dir` exists, handling both command-line arguments and YAML configuration."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, nargs="*",
                        help="Path(s) to YAML config file(s)")
    parser.add_argument("--trainer.logger.save_dir",
                        type=str, help="Logger save directory")
    args, _ = parser.parse_known_args(sys.argv[1:])

    save_dir = None  # Default to None

    if args.config:
        for config_path in args.config:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding='utf-8') as f:
                    try:
                        config = yaml.safe_load(f)
                        if config is not None:
                            # Safely navigate to trainer.logger.save_dir
                            trainer = config.get("trainer", {})
                            logger = trainer.get("logger", {})
                            if isinstance(logger, dict) :  # Ensure logger is a dictionary
                                yaml_save_dir = logger.get(
                                    "init_args", {}).get("save_dir")
                                if yaml_save_dir:
                                    save_dir = yaml_save_dir  # Use the first valid save_dir found
                                    break
                    except yaml.YAMLError as e:
                        print(f"Error parsing YAML file {config_path}: {e}")

    for i, arg in enumerate(sys.argv):
        if arg == "--trainer.logger.save_dir":
            save_dir = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
            break

    if not save_dir:
        print("Logger save_dir is None. No action taken.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Pre-created logger save_dir: {save_dir}")


class CustomSaveConfigCallback(SaveConfigCallback):
    '''save the config file to the logger's run directory, merge tags from different configs'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merged_tags = self._collect_tags_from_configs()

    def _collect_tags_from_configs(self):
        config_files = []
        merged_tags = set()

        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_files.append(sys.argv[i + 1])

        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        if isinstance(config_data, dict):
                            logger = config_data.get('trainer', {}).get(
                                'logger', {})
                            if logger and isinstance(logger, dict):
                                tags = logger.get('init_args', {}).get('tags', [])
                                if isinstance(tags, list):
                                    merged_tags.update(tags)
                except (yaml.YAMLError, IOError) as e:
                    print(f"Warning: Error reading {config_file}: {str(e)}")
        return merged_tags

    def setup(self, trainer, pl_module, stage):
        if hasattr(self.config, 'trainer') and hasattr(self.config.trainer, 'logger'):
            logger_config = self.config.trainer.logger
            if hasattr(logger_config, 'init_args'):
                logger_config.init_args['tags'] = list(self.merged_tags)
                if hasattr(trainer, 'logger') and trainer.logger is not None:
                    trainer.logger.experiment.tags = list(self.merged_tags)

        super().setup(trainer, pl_module, stage)

    def save_config(self, trainer, pl_module, stage) -> None:
        """Save the configuration file under the logger's run directory."""
        if stage == "predict":
            print("Skipping saving configuration in predict mode.")
            return  
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            project_name = trainer.logger.experiment.project_name()
            run_id = trainer.logger.experiment.id
            save_dir = trainer.logger.save_dir
            run_dir = os.path.join(save_dir, project_name, run_id)
            
            os.makedirs(run_dir, exist_ok=True)
            config_path = os.path.join(run_dir, "config.yaml")
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            print(f"Configuration saved to {config_path}")


class CustomWriter(BasePredictionWriter):
    """
    A custom prediction writer to save reconstructions to disk.
    """

    def __init__(self, output_dir: Path, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.outputs = defaultdict(list)
        
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        """
        Collect predictions batch by batch and organize them by volume.
        Assumes `predictions` contains a dictionary with 'volume_id' and 'slice_prediction'.
        """
        pass
        
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):

        gathered = [None] * torch.distributed.get_world_size()
        gathered_indices = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, predictions)
        torch.distributed.all_gather_object(gathered_indices, batch_indices)
        torch.distributed.barrier()
        if not trainer.is_global_zero:
            return
        predictions = sum(gathered, [])
        batch_indices = sum(gathered_indices, [])
        batch_indices = list(chain.from_iterable(batch_indices))
        outputs = defaultdict(list)
        num_slc_dict = {} # for reshape
        # Iterate through batches
        for batch_predictions in predictions:
            for i in range(len(batch_predictions["fname"])):
                fname = batch_predictions["fname"][i]
                slice_num = int(batch_predictions["slice_num"][i])
                output = batch_predictions["output"][i:i+1]
                outputs[fname].append((slice_num, output))
                # if num_slc_list[fname] exist, assign
                num_slc = batch_predictions["num_slc"][i].numpy()
                if fname not in num_slc_dict and num_slc!=-1:
                    num_slc_dict[fname] = batch_predictions["num_slc"][i]
        
        # Sort slices and stack them into volumes
        for fname in outputs:
            outputs[fname] = np.concatenate(
                [out.cpu() for _, out in sorted(outputs[fname])])
    
        # # Save the reconstructions
        save_reconstructions(outputs, num_slc_dict, self.output_dir / "reconstructions")
        print(f"Done! Reconstructions saved to {self.output_dir / 'reconstructions'}")


class CustomLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def instantiate_classes(self):
        super().instantiate_classes()
        if hasattr(self, 'config_init') and hasattr(self.config_init, 'subcommand') and self.config_init.subcommand == 'predict':
            self.model = PromptMrModule.load_from_checkpoint(self.config_init.predict.ckpt_path)
        
        # Patch the datamodule's dataloaders to use custom collate
        self._patch_datamodule()
    
    def _patch_datamodule(self):
        """Patch the datamodule to use custom collate function"""
        if not hasattr(self, "datamodule") or self.datamodule is None:
            return
            
        # Store original methods
        if hasattr(self.datamodule, 'train_dataloader'):
            original_train_dataloader = self.datamodule.train_dataloader
            
            # Create patched train_dataloader method
            def patched_train_dataloader(self):
                # Get the original dataloader
                dataloader = original_train_dataloader()
                if hasattr(dataloader, 'collate_fn') and dataloader.collate_fn == custom_collate:
                    # Already patched
                    return dataloader
                    
                # Create a new dataloader with our custom collate function
                new_dataloader = DataLoader(
                    dataloader.dataset,
                    batch_size=dataloader.batch_size,
                    shuffle=hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler),
                    num_workers=dataloader.num_workers,
                    pin_memory=getattr(dataloader, 'pin_memory', False),
                    drop_last=getattr(dataloader, 'drop_last', False),
                    collate_fn=custom_collate,
                )
                return new_dataloader
            
            # Apply the patch
            self.datamodule.train_dataloader = types.MethodType(patched_train_dataloader, self.datamodule)
        
        # Patch val_dataloader if it exists
        if hasattr(self.datamodule, 'val_dataloader'):
            original_val_dataloader = self.datamodule.val_dataloader
            
            # Create patched val_dataloader method
            def patched_val_dataloader(self):
                # Get the original dataloader
                dataloader = original_val_dataloader()
                if hasattr(dataloader, 'collate_fn') and dataloader.collate_fn == custom_collate:
                    # Already patched
                    return dataloader
                    
                # Create a new dataloader with our custom collate function
                new_dataloader = DataLoader(
                    dataloader.dataset,
                    batch_size=dataloader.batch_size,
                    shuffle=False,  # Don't shuffle validation
                    num_workers=dataloader.num_workers,
                    pin_memory=getattr(dataloader, 'pin_memory', False),
                    drop_last=getattr(dataloader, 'drop_last', False),
                    collate_fn=custom_collate,
                )
                return new_dataloader
            
            # Apply the patch
            self.datamodule.val_dataloader = types.MethodType(patched_val_dataloader, self.datamodule)


def run_cli():
    preprocess_save_dir()

    cli = CustomLightningCLI(
        save_config_callback=CustomSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    run_cli()