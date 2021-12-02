import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.notebook import trange, tqdm

# Ref: https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch/notebook
# All of the loss functions are from this kaggle kernel. It was hugely helpful for training.
# Loss functions and nn.modules:


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    """computes iou for one ground truth mask and predicted mask"""
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    """computes mean iou for a batch of ground truth masks and predicted masks"""
    ious = []
    preds = np.copy(outputs)  # copy is imp
    labels = np.array(labels)  # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


def dice_loss(mask, target):
    mask = torch.sigmoid(mask)
    smooth = 1.0
    iflat = mask.view(-1)
    tflat = target.contiguous().view(-1)  # Added contiguous
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, mask, target):
        if not (target.size() == mask.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), mask.size()))
        max_val = (-mask).clamp(min=0)
        loss = mask - mask * target + max_val + \
            ((-max_val).exp() + (-mask - max_val).exp()).log()  # tab once if broken
        invprobs = F.logsigmoid(-mask * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, mask, target):
        loss = self.alpha * self.focal(mask, target) - torch.log(dice_loss(mask, target))
        return loss.mean()


# Training and eval functions:
def training(path, model, dataset, learning_rate, epochs, batch_size=1, num_workers=0, device='cuda'):
    # Pass the dataset to a DataLoader
    cell_dl = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    num_batches = len(cell_dl)
    # Init loss and optimizer
    criterion = MixedLoss(10.0, 2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    epoch_bar = trange(epochs, desc=f'Epoch: {0}/{epochs} - Progress')
    for epoch in epoch_bar:
        running_loss = 0.0
        optimizer.zero_grad()
        load_batch = tqdm(cell_dl, desc=f'Batch: {0}/{num_batches} - Progress', leave=False)
        for i, batch in enumerate(load_batch):
            # Load and process img and mask
            img_id, images, masks = batch
            masks = torch.permute(masks, (0, 3, 2, 1)).to(device)
            images = torch.permute(images, (0, 3, 2, 1)).to(device)
            # init output holder
            output = torch.zeros(images.shape).to(device)

            # Forward Pass
            with torch.cuda.amp.autocast():
                output[:][0:2] = model(images)[0][0]  # Cast Height and Width dims to all 3 output_channels
                loss = criterion(output, masks)
            load_batch.set_description(f'Batch: {i}/{num_batches} - Loss: {loss.item():.3f} - Progress')

            # Backward Pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        epoch_bar.set_description(
            f'Epoch: {epoch}/{epochs} - Epoch-Loss: {(running_loss / num_batches):.3f} - Progress')
        epoch_bar.refresh()
    # Save model state dict for loading later
    # Uncomment to save
    torch.save(model.state_dict(), f'{path}/model/model.pth')
    torch.cuda.empty_cache()
    return model


def evaluate(model, dataset, device):
    # Init loss and test dataloader
    criterion = MixedLoss(10.0, 2.0)
    test_dl = DataLoader(dataset=dataset, batch_size=1, drop_last=True, shuffle=True)
    eval_bar = tqdm(test_dl, desc=f'Eval: {0}/{len(test_dl) * test_dl.batch_size} - Progress')
    running_loss = 0.0
    num_batches = float(len(test_dl))

    # Loop over testing dataset
    for i, batch in enumerate(eval_bar):
        img_id, images, masks = batch
        masks = torch.permute(masks, (0, 3, 2, 1)).to(device)
        images = torch.permute(images, (0, 3, 2, 1)).to(device)
        output = torch.zeros(images.shape).to(device)
        # Forward pass
        with torch.no_grad():
            output[:][0:2] = model(images)[0][0]
            loss = criterion(output, masks)
            running_loss += loss.item()
        eval_bar.set_description(f'Eval: {i}/{len(test_dl) * test_dl.batch_size} - Loss: {loss.item():.3f} - Progress')
        eval_bar.refresh()
    return running_loss / num_batches


def show_predictions(model, dataset, device):
    # Load data
    test_dl = DataLoader(dataset=dataset, batch_size=1, drop_last=True, shuffle=True)
    loaded = iter(test_dl)
    criterion = MixedLoss(10.0, 2.0)
    counter = 0
    # Plot loop
    print('\nPredictions')
    print('-----------------------------------------\n')
    for batch in loaded:
        # Prep batch
        img_id, image, mask = batch
        mask = torch.permute(mask, (0, 3, 2, 1)).to(device)
        image = torch.permute(image, (0, 3, 2, 1)).to(device)
        output = torch.zeros(image.shape).to(device)
        loss = 0.0
        # Forward pass
        with torch.no_grad():
            output[:][0:2] = model(image)[0][0]
            loss += criterion(output, mask).item()
        # Threshold output
        output = torch.sigmoid(output)
        output[output > 0.6] = 1.0
        output[output < 0.4] = 0.0
        # Unpermute inputs/outputs and plot
        image = torch.permute(image, (0, 2, 3, 1))
        mask = torch.permute(mask, (0, 2, 3, 1))
        output = torch.permute(output, (0, 2, 3, 1))
        print(f'\nImage: {img_id[0]}')
        plt.matshow(image[0].to('cpu').numpy())
        plt.show()
        print(f'Predicted Mask: {img_id[0]} - Loss: {loss:.3f}')
        plt.matshow(output[0].to('cpu').numpy())
        plt.show()
        print(f'Ground Truth Mask: {img_id[0]}')
        plt.matshow(mask[0].to('cpu').numpy())
        plt.show()
        print('-----------------------------------------\n')
        # Probably a better way to do this but subplot is weird with matshow and I don't like imshow
        counter += 1
        if counter == 3:
            break
