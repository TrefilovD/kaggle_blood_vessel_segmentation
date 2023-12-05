import os
from argparse import ArgumentParser

from clearml import Task, Logger
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from computing_loops import train_and_test
from consts import *
from config import parse_config
from dataset import CustomDataset, augment_image
from loss import get_loss
from model import get_model
from init_optimizer import get_optimizer


def parse_config_path():
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        default=r"kaggle_blood_vessel_segmentation\configs\baseline.yaml",
    )
    args = parser.parse_args()
    return args.path

if __name__ == "__main__":
    cfg = parse_config(parse_config_path())

    if cfg.clearml.use_clearml:
        task = Task.init(
            project_name='KaggleBloodVesselSegmentation',
            task_name=cfg.clearml.name_task,
            tags=cfg.clearml.tags
        )
        logger = Logger.current_logger()
        # TODO сделать вариации датасетов и сделать загрузку в ClearML
    else:
        task = None
        logger = None

    dataset = 'kidney_1_dense'
    if task:
        task.upload_artifact(name='data.choosed_dataset', artifact_object=dataset)

    images_path = os.path.join(BASE_DIR, "train", dataset, 'images')
    labels_path = os.path.join(BASE_DIR, "train", dataset, 'labels')

    image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.tif')])
    label_files = sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.tif')])

    train_image_files, val_image_files, train_mask_files, val_mask_files = train_test_split(
        image_files, label_files,
        test_size=cfg.dataset.test_size,
        random_state=cfg.env.seed
    )

    train_dataset = CustomDataset(train_image_files, train_mask_files, augmentation_transforms=augment_image)
    val_dataset = CustomDataset(val_image_files, val_mask_files, augmentation_transforms=augment_image)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.loops_settings.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.loops_settings.batch_size, shuffle=False)

    model = get_model(cfg.model, cfg.model_settings, task)
    optimizer = get_optimizer(cfg.optimizer, cfg.optimizer_settings, model.parameters(), task)
    criterion = get_loss(cfg.loss, cfg.loss_settings, task)
    
    train_and_test()
