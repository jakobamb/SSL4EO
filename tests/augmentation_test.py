import shutil
from pathlib import Path

import pytest
from torchvision import transforms as T

from data import constants
from data.constants import MMEARTH_DIR
from data.mmearth_dataset import MMEarthDataset, create_MMEearth_args, get_mmearth_dataloaders
from methods import transforms
from methods.transforms import to_tensor

input_size = 112


@pytest.mark.parametrize(
    "transform",
    [
        transforms.BarlowTwinsTransform(
            transforms.BarlowTwinsView1Transform(input_size=input_size),
            transforms.BarlowTwinsView2Transform(input_size=input_size),
        ),
        transforms.SimCLRTransform(input_size=input_size),
        transforms.BYOLTransform(
            transforms.BYOLView1Transform(input_size=input_size),
            transforms.BYOLView2Transform(input_size=input_size),
        ),
        transforms.VICRegTransform(input_size=input_size),
        transforms.MAETransform(input_size=input_size),
    ],
)
def test_augmentations(transform):
    modalities = constants.INP_MODALITIES
    split = "train"
    args = create_MMEearth_args(MMEARTH_DIR, modalities, constants.MODALITIES_FULL)

    args.modalities = modalities
    transform = T.Compose([to_tensor, transform])
    dataset = MMEarthDataset(args, split=split, transform=transform)

    if split == "train":
        num_samples = 10
        for i, data in enumerate(dataset):
            if i >= num_samples:
                break

@pytest.mark.parametrize(
    "transform",
    [
        transforms.BarlowTwinsTransform(
            transforms.BarlowTwinsView1Transform(input_size=input_size),
            transforms.BarlowTwinsView2Transform(input_size=input_size),
        ),
        transforms.SimCLRTransform(input_size=input_size),
        transforms.BYOLTransform(
            transforms.BYOLView1Transform(input_size=input_size),
            transforms.BYOLView2Transform(input_size=input_size),
        ),
        transforms.VICRegTransform(input_size=input_size),
        transforms.MAETransform(input_size=input_size),
    ],
)
@pytest.mark.parametrize(
    "no_ffcv", [False, True],
)
def test_augmentation_dataloader(transform, no_ffcv):
    modalities = constants.INP_MODALITIES

    test_out = Path("test_out")
    test_out.mkdir(exist_ok=True)
    target_modality = {"biome": constants.MODALITIES_FULL["biome"]}

    try:
        loader = get_mmearth_dataloaders(
            transform, constants.MMEARTH_DIR, test_out,
            modalities, target_modality, 4, 64, ["train"], no_ffcv,
        )

        num_batches = 10
        for b_i, data in enumerate(loader):
            if b_i >= num_batches: break
    finally:
        # cleanup
        shutil.rmtree(test_out, ignore_errors=True)