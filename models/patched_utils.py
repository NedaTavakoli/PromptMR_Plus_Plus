# patched_utils.py
from models.utils import KSpaceACSExtractor as OriginalKSpaceACSExtractor

class PatchedKSpaceACSExtractor(OriginalKSpaceACSExtractor):
    def __call__(self, masked_kspace, mask, num_low_frequencies, mask_type):
        """
        Extract the center (ACS) region of the k-space.
        
        Args:
            masked_kspace: Masked k-space data of shape (batch_size, num_coils, height, width, complex=2)
            mask: Sampling mask
            num_low_frequencies: Number of low frequency lines for center extraction
            mask_type: Type of mask (cartesian, poisson_disc, or others)
            
        Returns:
            Center (ACS) of the k-space data of shape (batch_size, num_coils, height, width_ACS, complex=2)
        """
        # Accept any mask_type, but normalize it to one of the expected values
        if mask_type is None:
            mask_type = "cartesian"  # Default to cartesian
        
        mask_type_lower = str(mask_type).lower()
        
        # Force to a valid value
        if mask_type_lower in ["cartesian", "cartes", "cart"]:
            # Handle cartesian mask
            return self._extract_acs_cartesian(masked_kspace, num_low_frequencies)
        elif mask_type_lower in ["poisson_disc", "poisson-disc", "poisson disc", "poisson"]:
            # Handle poisson disc mask
            return self._extract_acs_poisson(masked_kspace)
        else:
            # For any other mask type, default to cartesian handling
            print(f"[WARNING] Unrecognized mask_type '{mask_type}', defaulting to cartesian handling")
            return self._extract_acs_cartesian(masked_kspace, num_low_frequencies)