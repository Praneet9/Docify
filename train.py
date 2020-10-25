import numpy as np
from tqdm import tqdm
import cv2

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch import optim
import torch.nn as nn

from utils.unet import UNet
from utils.dataset import BasicDataset


def train_model(model, device, 
                img_dir, mask_dir, 
                checkpoint_dir, 
                checkpoint_file=None, 
                epochs=20, lr=0.001, 
                val_split=0.20, 
                batch_size=1):
    
    dataset = BasicDataset(img_dir, mask_dir)
    val_samples = int(len(dataset) * val_split)
    train_samples = len(dataset) - val_samples
    train, val = random_split(dataset, [train_samples, val_samples])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    
    writer = SummaryWriter(log_dir=checkpoint_dir, comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    criterion = nn.BCEWithLogitsLoss()
    
    training_loss = []
    validation_loss = []
    current_epoch = 0
    
    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_dir + checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch']
        training_loss = checkpoint['loss']
        validation_loss = checkpoint['val_loss']
        global_step = checkpoint['global_step']

    for epoch in range(1 + current_epoch, epochs + 1):
        model.train()

        losses = []
        val_losses = []
        avg_val_loss = np.inf
        
        with tqdm(total=train_samples, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if model.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                
                masks_pred = model(imgs)
                loss = criterion(masks_pred, true_masks)
                losses.append(loss.item())
                writer.add_scalar('Loss/train', sum(losses)/len(losses), global_step)

                pbar.set_postfix(**{'loss': sum(losses)/len(losses)})
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                
                pbar.update(imgs.shape[0])
                global_step += 1

            val_loss = 0
            for val_batch in val_loader:
                imgs, true_masks = val_batch['image'], val_batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    mask_pred = model(imgs)
                
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                val_loss += criterion(masks_pred, true_masks).item()
            val_score = val_loss / len(val_loader)
            val_losses.append(val_score)
            avg_val_loss = sum(val_losses) / len(val_losses)
            pbar.set_postfix(**{'loss': sum(losses)/len(losses), 'val_loss': avg_val_loss})
        
        training_loss.append(sum(losses)/len(losses))
        validation_loss.append(avg_val_loss)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': training_loss,
            'val_loss': validation_loss,
            'global_step': global_step
        }, checkpoint_dir + str(epoch) + '_model.pth')
        
    writer.close()

if __name__ == '__main__':
    
    IMAGES_PATH = '/data/Data/midv500_data/dataset/images_resized/'
    MASKS_PATH = '/data/Data/midv500_data/dataset/masks_resized/'
    MODEL_CHECKPOINT_PATH = '/data/Data/midv500_data/dataset/checkpoints/'
    
    dataset = BasicDataset(IMAGES_PATH, MASKS_PATH)
    
    unet = UNet(n_channels=3, n_classes=1)
    summary(unet.cuda(), (3, 480, 360))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    unet = unet.to(device=device)
    
    train_model(unet,
            device,
            IMAGES_PATH,
            MASKS_PATH,
            MODEL_CHECKPOINT_PATH)