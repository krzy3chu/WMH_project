import torch
import cv2
import numpy as np
from training import segmentation_module, data_module, model_name


idx = 34

data_module.setup('test')
image, mask = data_module.val_dataset[idx]

model = segmentation_module
model.load_state_dict(torch.load(model_name))
model.eval().to('cpu')

with torch.no_grad():
    result = torch.sigmoid(model(image[None, ...]))

print(np.amax(result.numpy()))
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