import cv2
import numpy as np
import nibabel as nib

from pathlib import Path
from tqdm import tqdm


raw_data = Path('./data/raw')
processed_data = Path('./data/processed')


def load_raw_volume(path: Path) -> np.ndarray:
    data: nib.Nifti1Image = nib.load(path)
    data = nib.as_closest_canonical(data)  # set scan axis orientation
    raw = data.get_fdata(caching='unchanged', dtype=np.float32)
    return raw


def load_labels_volume(path: Path) -> np.ndarray:
    labels = load_raw_volume(path)
    labels[labels > 1] = 0  # ignore labels higher than 1
    return labels


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume -= np.min(volume)
    volume /= np.max(volume)
    return (volume * 255).astype(np.uint8)


# iterate over every element in database, normalize each one as volume and extract to slices and masks
for brain in tqdm(raw_data.iterdir(), desc="Processing data", total=len(list(raw_data.iterdir()))):
    for scan in brain.glob('*.nii.gz'):
        if 'labels_wmh' in str(scan):
            volume = load_labels_volume(scan)
        else:
            volume = load_raw_volume(scan)
        volume = normalize_volume(volume)
        for z in range(volume.shape[2]):
            filepath = Path(processed_data, scan.stem.split('.')[0])
            filepath.mkdir(parents=True, exist_ok=True)
            filename = scan.parent.stem + '_' + str(z) + '.png'
            cv2.imwrite(str(filepath / filename), volume[:, :, z])