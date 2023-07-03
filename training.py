from segmentation import *


database = Path('./data/processed')

segmentation_module = SegmentationModule()
data_module = SegmentationDataModule(data_dir=database)

trainer = pl.Trainer(
    logger=True,
    max_epochs=1, 
    )

if __name__ == "__main__":

    trainer.fit(
        model=segmentation_module,
        datamodule=data_module
        )