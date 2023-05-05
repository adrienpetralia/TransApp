#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia
# @description : Geometric Mask and Masked MSE Loss for Pretraining
# @component: src/utils/
# @file : losses.py
#
#################################################################################################################

import torch
import numpy as np
    
class GeometricMask(torch.nn.Module):
    """ 
    Geometric Mask
    """
    def __init__(self, mean_length=3, masking_ratio=0.15, type_corrupt='zero', value=-10, dim_masked=0):
        super(GeometricMask, self).__init__()
        self.masking_ratio = masking_ratio
        self.mean_length = mean_length
        self.type_corrupt = type_corrupt
        self.value = value
        self.dim_masked = dim_masked
    
    def forward(self, x):
        submask = np.ones(x.shape[-1], dtype=bool)
        p_m = 1 / self.mean_length 
        p_u = p_m * self.masking_ratio / (1 - self.masking_ratio) 
        p = [p_m, p_u]
        
        state = int(np.random.rand() > self.masking_ratio)
        for i in range(x.shape[-1]):
            submask[i] = state
            if np.random.rand() < p[state]:
                state = 1 - state

        mask = np.ones(x.shape)
        mask[:, self.dim_masked, :] = submask
        mask = (~torch.Tensor(mask).bool())
        
        if self.type_corrupt=='noise':
            gaussian_noise = torch.mul(torch.randn(x.shape), mask)
            x_masked = torch.add(x, gaussian_noise)            
        elif self.type_corrupt=='zero':
            x_masked = torch.mul(x, ~mask)
        elif self.type_corrupt=='value':
            tensor = torch.mul(torch.Tensor([self.value]).repeat(x.shape), mask)
            x_masked = torch.add(torch.mul(x, ~mask), tensor)
        
        return mask, x_masked

# MaskedMSELoss from MTST repository : https://github.com/gzerveas/mvts_transformer/blob/3f2e378bc77d02e82a44671f20cf15bc7761671a/src/models/loss.py
class MaskedMSELoss(torch.nn.Module):
    """ 
    Masked MSE Loss
    """
    def __init__(self, type_loss: str='MSE', reduction: str='mean', masked_dimension=0):

        super().__init__()

        self.reduction = reduction
        self.masked_dimension = masked_dimension
        
        if type_loss=='L1':
            self.mse_loss = torch.nn.L1Loss(reduction=self.reduction)
        else:
            self.mse_loss = torch.nn.MSELoss(reduction=self.reduction)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between a target value and a prediction.
        
        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered
        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask.bool())
        masked_true = torch.masked_select(y_true, mask.bool())

        return self.mse_loss(masked_pred, masked_true)

