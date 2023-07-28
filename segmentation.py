import torch
import cv2

import albumentations as A
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pathlib import Path
from typing import List, Iterator
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler, SequentialSampler
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score
from monai.losses.dice import DiceLoss


class SegmentationDataset(Dataset):
    def __init__(self, data_dir: Path, slices_names: list[str], transforms: A.Compose):
        self._images = [data_dir / 'flair' / slice_name
                        for slice_name in slices_names]
        self._labels = [data_dir / 'labels_wmh' / slice_name
                        for slice_name in slices_names]
        self._transforms = transforms

    def __getitem__(self, index: int):
        image_path = self._images[index]
        labels_path = self._labels[index]

        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        labels = cv2.imread(str(labels_path), cv2.IMREAD_GRAYSCALE) / 255

        transformed = self._transforms(image=image, mask=labels)

        return transformed['image'], transformed['mask'][None, ...].type(torch.float32)

    def __len__(self):
        return len(self._images)
    

class SegmentationBatchSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int = 16):
        self.dataset = dataset
        self.batch_size = batch_size

        self.labeled = []
        self.empty = []
        
        for idx in SequentialSampler(dataset):  # divide indexes into two subsets
            (self.labeled if 1 in dataset[idx][1] else self.empty).append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        batch = [0] * self.batch_size
        idx_in_batch = 0
        for idx in range(len(self.labeled)):
            batch[2 * idx_in_batch] = self.labeled[idx]
            batch[2 * idx_in_batch + 1] = self.empty[idx]
            idx_in_batch += 1
            if 2 * idx_in_batch == self.batch_size:
                yield batch
                idx_in_batch = 0
                batch = [0] * self.batch_size
        if idx_in_batch > 0:
            yield batch[:2 * idx_in_batch]

    def __len__(self):  # returns number of batches in one epoch
        return (2 * len(self.labeled) + self.batch_size - 1) // self.batch_size
    

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 16):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

        self._augmentations = A.Compose([
            A.Resize(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ToFloat(max_value=255),
            ToTensorV2()
        ])
        self._transforms = A.Compose([
            A.Resize(width=256, height=256),
            A.ToFloat(max_value=255),
            ToTensorV2()
        ])

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str):
        slices_names = sorted(
            [slice_path.name
             for slice_path in (self.data_dir / 'labels_wmh').iterdir()])

        train_slices, val_slices = train_test_split(
            slices_names, test_size=0.15, random_state=42)

        self.train_dataset = SegmentationDataset(self.data_dir, train_slices,
                                                self._augmentations)
        self.val_dataset = SegmentationDataset(self.data_dir, val_slices,
                                                self._transforms)

    def train_dataloader(self):
        batch_sampler = SegmentationBatchSampler(self.train_dataset, self.batch_size)
        return  DataLoader(self.train_dataset, batch_sampler=batch_sampler, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)
    

class SegmentationModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.network = smp.Unet('resnet18')
        # self.loss_function = nn.BCEWithLogitsLoss()
        self.loss_function = DiceLoss()

        metrics = MetricCollection([
            BinaryAccuracy(),
            BinaryF1Score(),
            BinaryRecall(),
            BinaryPrecision()
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)

        metric = self.train_metrics(y_pred, y)
        self.log_dict(metric)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        metric = self.valid_metrics(y_pred, y)
        self.log_dict(metric)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)