import time

from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from clearml import Task, Logger
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm, trange

from loss import dice_coeff
from utils import show_images as draw


def train_and_test(
        model: nn.Module,
        dataloaders: DataLoader,
        optimizer: Optimizer,
        criterion: nn.modules.loss._Loss,
        device: Literal["cpu", "cuda"],
        num_epochs: int = 100,
        show_images: bool = False,
        task: Optional[Task] = None,
        logger: Optional[Logger] = None
    ):
    since = time.time()
    best_loss = 1e10
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fieldnames = ['epoch', 'training_loss', 'test_loss', 'training_dice_coeff', 'test_dice_coeff']
    train_epoch_losses = []
    test_epoch_losses = []

    pbar = trange(range(1, num_epochs + 1), desc="Training...")
    for epoch in pbar(range(1, num_epochs + 1)):
        pbar.set_description(f'Epoch {epoch}/{num_epochs}')
        
        batchsummary = {a: [0] for a in fieldnames}
        batch_train_loss = 0.0
        batch_test_loss = 0.0

        for phase in ['training', 'test']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            for sample in iter(dataloaders[phase]):

                inputs = sample[0].to(device)
                masks = sample[1].to(device)
                if show_images:
                    draw([inputs, masks], ["Image", "Mask"])
                
                masks = masks.unsqueeze(1)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)

                    loss = criterion(outputs, masks)

                    y_pred = outputs.data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()

                    batchsummary[f'{phase}_dice_coeff'].append(dice_coeff(y_pred, y_true))

                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                        batch_train_loss += loss.item() * sample[0].size(0)

                    else:
                        batch_test_loss += loss.item() * sample[0].size(0)

            if phase == 'training':
                epoch_train_loss = batch_train_loss / len(dataloaders['training'])
                train_epoch_losses.append(epoch_train_loss)
            else:
                epoch_test_loss = batch_test_loss / len(dataloaders['test'])
                test_epoch_losses.append(epoch_test_loss)

            batchsummary['epoch'] = epoch
            
            info = '{} Loss: {:.4f}'.format(phase, loss)
            pbar.write(info)

        best_loss = np.max(batchsummary['test_dice_coeff'])
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        info = \
            f'\t\t\t train_dice_coeff: {batchsummary["training_dice_coeff"]}, test_dice_coeff: {batchsummary["test_dice_coeff"]}'
        pbar.write(info)

    info = 'Best dice coefficient: {:4f}'.format(best_loss)
    print(info)

    return model, train_epoch_losses, test_epoch_losses