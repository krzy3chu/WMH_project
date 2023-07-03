import torch
import cv2

from pathlib import Path
from training import segmentation_module, data_module


idx = 34
model_checkpoint = Path("lightning_logs/version_5/checkpoints/epoch=0-step=148.ckpt")

data_module.setup('test')
image, mask = data_module.val_dataset[idx]

model = segmentation_module.load_from_checkpoint(model_checkpoint)
model.eval().to('cpu')

with torch.no_grad():
    result = torch.sigmoid(model(image[None, ...]))

# THRESHOLD = 0.5
# result[result<THRESHOLD] = 0
# result[result>=THRESHOLD] = 1

while True:
    cv2.imshow('Scan', image.permute(1, 2, 0).numpy())
    cv2.imshow('Mask', mask.squeeze().numpy())
    cv2.imshow('Predicted result', result.squeeze().numpy())

    key_code = cv2.waitKey(10)
    if key_code == 27:  # escape key
        break