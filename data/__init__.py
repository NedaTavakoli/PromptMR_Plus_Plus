
# data/__init__.py
from .cmr2025 import CMR2025Dataset, CMR2025DataModule

__all__ = ["CMR2025Dataset", "CMR2025DataModule"]
from .mri_data import (
    RawDataSample,
    BalanceSampler,
    FuncFilterString,
    CombinedSliceDataset,
    CalgaryCampinasSliceDataset,
    CmrxReconSliceDataset,
    CmrxReconInferenceSliceDataset,
    FastmriSliceDataset
)
from .transforms import (
    CalgaryCampinasDataTransform,
    FastmriDataTransform,
    CmrxReconDataTransform,
    to_tensor,
)
from .volume_sampler import (
    VolumeSampler,
    InferVolumeDistributedSampler,
    InferVolumeBatchSampler
)
from .subsample import (
    PoissonDiscMaskFunc,
    FixedLowEquiSpacedMaskFunc,
    RandomMaskFunc,
    EquispacedMaskFractionFunc,
    FixedLowRandomMaskFunc,
    CmrxRecon24MaskFunc,
    CmrxRecon24TestValMaskFunc
)

def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=True,
        collate_fn=PromptMrModule.custom_collate,  # Use the custom collate function
    )

def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=True,
        collate_fn=PromptMrModule.custom_collate,  # Use the custom collate function
    )