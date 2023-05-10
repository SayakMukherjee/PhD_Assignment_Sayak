#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-May-2023
#
# References:
# 1. solo-learn: https://github.com/vturrisi/solo-learn/tree/main/solo/losses
# ---------------------------------------------------------------------------
# Custom loss functions used in training
# ---------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as T

class PSELoss(torch.nn.Module):

    def __init__(self, kernel_size, sigma):
        super(PSELoss, self).__init__()

        self.kernel = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, recons: torch.Tensor, orig: torch.Tensor) -> torch.Tensor:

        residuals = recons - orig
        return (self.kernel(residuals)).sum(dim=1).mean()


class ContractiveLoss(torch.nn.Module):

    def __init__(self):
        super(ContractiveLoss, self).__init__()

    def forward(self, z: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:

        z.backward(torch.ones(z.size()).to(z.device), retain_graph=True)

        loss = torch.sqrt(torch.sum(torch.pow(inputs.grad, 2)))

        inputs.grad.data.zero_() # remove the gradient data

        return loss

class BYOLLoss(torch.nn.Module):

    def __init__(self, simplified: bool = True):
        super(BYOLLoss, self).__init__()

        self.simplified = simplified

    def forward(self, p: torch.Tensor, z: torch.Tensor, indexes: torch.Tensor = None) -> torch.Tensor:
        """Computes BYOL's loss given batch of predicted features p and projected momentum features z.
        Args:
            p (torch.Tensor): NxD Tensor containing predicted features from view 1
            z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
            simplified (bool): faster computation, but with same result. Defaults to True.
            indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
                or targets of each crop (supervised).
        Returns:
            torch.Tensor: BYOL's loss.
        """

        if self.simplified:
            return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean()

        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)

        return 2 - 2 * (p * z.detach()).sum(dim=1).mean()

class SimSiamLoss(torch.nn.Module):

    def __init__(self, simplified: bool = True):
        super(SimSiamLoss, self).__init__()

        self.simplified = simplified

    def forward(self, p: torch.Tensor, z: torch.Tensor, indexes: torch.Tensor = None) -> torch.Tensor:
        """Computes SimSiam's loss given batch of predicted features p from view 1 and
        a batch of projected features z from view 2.
        Args:
            p (torch.Tensor): Tensor containing predicted features from view 1.
            z (torch.Tensor): Tensor containing projected features from view 2.
            simplified (bool): faster computation, but with same result.
            indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
                or targets of each crop (supervised).
        Returns:
            torch.Tensor: SimSiam loss.
        """

        if self.simplified:
            return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)

        return -(p * z.detach()).sum(dim=1).mean()

class SimCLRLoss(torch.nn.Module):

    def __init__(self, temperature: float = 0.1):
        super(SimCLRLoss, self).__init__()

        self.temperature = temperature

    def forward(self, z: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
        """Computes SimCLR's loss given batch of projected features z
        from different views, a positive boolean mask of all positives and
        a negative boolean mask of all negatives.
        Args:
            z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
            indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
                or targets of each crop (supervised).
        Return:
            torch.Tensor: SimCLR loss.
        """

        z = F.normalize(z, dim=-1)
        gathered_z = gather(z)

        sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / self.temperature)

        gathered_indexes = gather(indexes)

        indexes = indexes.unsqueeze(0)
        gathered_indexes = gathered_indexes.unsqueeze(0)
        # positives
        pos_mask = indexes.t() == gathered_indexes
        pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)
        # negatives
        neg_mask = indexes.t() != gathered_indexes

        pos = torch.sum(sim * pos_mask, 1)
        neg = torch.sum(sim * neg_mask, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return loss

class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)

def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0