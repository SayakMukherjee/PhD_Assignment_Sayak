#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-May-2023
#
# References:
# 1. https://github.com/The-AI-Summer/byol-cifar10/blob/main/ai_summer_byol_in_cifar10.py
# 2. https://docs.lightly.ai/self-supervised-learning/examples/byol.html
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T

from typing import Callable, List, Optional, Type, Tuple, Union
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image

IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

class SaltPepperNoise(object):

    def __init__(self):

        # self.sp_prob = sp_prob
        pass

    def __call__(self, image: Union[Image.Image, torch.Tensor]):

        image_np = image.cpu().numpy()

        # TODO: Get values
        seed = 40
        amount = 0.1
        prop_salt_pepper = 0.5


        random_gen = np.random.default_rng(seed)

        selected_pixels = self._bernoulli(amount, image_np.shape, random_gen)
        salted = self._bernoulli(prop_salt_pepper, image_np.shape, random_gen)
        peppered = ~salted

        image_np[selected_pixels & salted] = 0.
        image_np[selected_pixels & peppered] = 1.

        image = torch.from_numpy(image_np).to(image.device)

        return image


    def _bernoulli(self, p, shape, random_state):
        """Bernoulli trails at a given probability

        Ref: https://github.com/scikit-image/scikit-image/blob/v0.20.0/skimage/util/noise.py#L39-L234

        Args:
            p (float): probability of returning `True`
            shape (int or tuple): Shape of ndarray to return
            random_state (numpy.random.Generator): generator instance

        Returns:
            _type_: _description_
        """
        if p == 0:
            return np.zeros(shape, dtype=bool)
        if p == 1:
            return np.ones(shape, dtype=bool)
        return random_state.random(shape) <= p

class RandomRotate:

    def __init__(self, rr_prob: float, rr_degrees= Union[None, float, Tuple[float, float]]):
        self.prob = rr_prob

        if rr_degrees is None:
            self.angle = 90
            self.transform = None
        else:
            self.angle = None
            self.transform = T.RandomApply([T.RandomRotation(degrees=rr_degrees)], p=rr_prob)

    def __call__(self, image: Union[Image.Image, torch.Tensor]):

        if self.transform is None:
            prob = np.random.random_sample()
            if prob < self.prob:
                image = TF.rotate(image, self.angle)
            return image
        else:
            return self.transform(image)

class CustomCollate(nn.Module):

    def __init__(
            self,
            input_size: int = 64,
            cj_prob: float = 0.8,
            cj_bright: float = 0.7,
            cj_contrast: float = 0.7,
            cj_sat: float = 0.7,
            cj_hue: float = 0.2,
            min_scale: float = 0.15,
            random_gray_scale: float = 0.2,
            gaussian_blur: float = 0.5,
            kernel_size: Optional[float] = None,
            sigma: Tuple[float, float] = (0.2, 2),
            vf_prob: float = 0.0,
            hf_prob: float = 0.5,
            rr_prob: float = 0.0,
            rr_degrees: Union[None, float, Tuple[float, float]] = None,
            normalize: dict = IMAGENET_NORMALIZE
        ):
        super(CustomCollate, self).__init__()

        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)
        blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

        transform = [
            # T.ToTensor(),
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            RandomRotate(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomApply([blur], p=gaussian_blur),
            T.RandomGrayscale(p=random_gray_scale)
        ]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]

        self.transform = T.Compose(transform)

    def forward(self, batch):

        batch_size = len(batch)

        # list of indexes
        indexes = torch.LongTensor([item[0] for item in batch])

        # list of transformed images
        transforms = [
            self.transform(batch[i % batch_size][1]).unsqueeze_(0)
            for i in range(2 * batch_size)
        ]

        # list of labels
        labels = torch.LongTensor([item[2] for item in batch])

        # tuple of transforms
        transforms = (
            torch.cat(transforms[:batch_size], 0),
            torch.cat(transforms[batch_size:], 0),
        )

        return indexes, transforms, labels

class SimCLRCollate(CustomCollate):

    def __init__(
        self,
        input_size: int = 32,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = 3,
        sigma: Tuple[float, float] = (0.1, 2.0),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: dict = IMAGENET_NORMALIZE,
    ):
        super(SimCLRCollate, self).__init__(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_bright=cj_strength * 0.8,
            cj_contrast=cj_strength * 0.8,
            cj_sat=cj_strength * 0.8,
            cj_hue=cj_strength * 0.2,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigma=sigma,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )