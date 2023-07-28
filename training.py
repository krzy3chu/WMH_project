from segmentation import *
from neptune_api import API_TOKEN
from pytorch_lightning.loggers import NeptuneLogger


database = Path('./data/processed')
model_name = Path('./WMH_model')

segmentation_module = SegmentationModule()
data_module = SegmentationDataModule(data_dir=database)

PROJECT_NAME = "kozaaaaa/WMH"

neptune_logger = NeptuneLogger(
    api_key=API_TOKEN,
    project=PROJECT_NAME,
    tags=["WMH"]
)

trainer = pl.Trainer(
    logger=neptune_logger,
    max_epochs=1,
    )

if __name__ == "__main__":

    trainer.fit(
        model=segmentation_module,
        datamodule=data_module
        )
    
    torch.save(segmentation_module.state_dict(), model_name)