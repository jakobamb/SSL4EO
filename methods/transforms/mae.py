from typing import Tuple, Union

import kornia.augmentation as K
from kornia.constants import Resample
from torch import Tensor
from torch import nn


class MAETransform(nn.Sequential):
    """Implements the view augmentation for MAE [0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 1.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377

    Attributes:
        input_size:
            Size of the input image in pixels.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.

    """

    def __init__(
        self, input_size: int = 112, min_scale: float = 0.2
    ):
        super().__init__(
            K.RandomResizedCrop(
                (input_size, input_size), scale=(min_scale, 1.0), resample=Resample.BICUBIC
            ),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(), # addition that is not in paper
        )
        self.input_size = input_size

    def forward(self, input: Tensor) -> list[Tensor]:
        return [super().forward(input)]


class MAE_AdditionalTransform(nn.Sequential):
    """Implements the view augmentation for MAE [0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 1.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377

    Attributes:
        input_size:
            Size of the input image in pixels.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.

    """

    def __init__(
        self, 
        input_size: int = 112, 
        min_scale: float = 0.2,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        solarize_prob: float = 0.1,
        sigmas: Tuple[float, float] = (0.1, 2),
    ):
        super().__init__(
            # below needs to be implemented to work on all 12 channels...
            # K.ColorJitter(
            #     brightness=cj_strength * cj_bright,
            #     contrast=cj_strength * cj_contrast,
            #     saturation=cj_strength * cj_sat,
            #     hue=cj_strength * cj_hue,
            #     p=cj_prob,
            # ),
            # K.RandomGrayscale(p=random_gray_scale),
            K.RandomSolarize(p=solarize_prob),
            K.RandomGaussianBlur(
                kernel_size=input_size // 10,
                sigma=sigmas,
                p=gaussian_blur,
                border_type="constant",
            ),
        )
        self.input_size = input_size

    def forward(self, input: Tensor) -> list[Tensor]:
        return [super().forward(input)]