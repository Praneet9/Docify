import numpy as np
from utils.unet import UNet

import torch
import cv2
from utils.image_aug import normalization2

def get_image_tensor(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (360, 480))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    og_image = image.copy()
    
    # Normalize the image
    image = normalization2(image, max=1, min=0)

    # HWC to CHW
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)

    image = torch.from_numpy(image).type(torch.FloatTensor)
    
    return og_image, image

def get_mask(model, device, image):
    image = image.to(device=device, dtype=torch.float32)
    
    mask_pred = model(image)
    pred = torch.sigmoid(mask_pred)
    pred = (pred > 0.5).float()

    pred = pred.squeeze()
    pred = pred.cpu().detach().numpy()
    
    return pred

if __name__ == '__main__':
    
    unet = UNet(n_channels=3, n_classes=1)
    checkpoint = torch.load('17_model.pth')
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    unet = unet.to(device=device)
    
    img_path = '/data/Data/midv500_data/IMG_2126.JPG'
    
    original, img_tensor = get_image_tensor(img_path)
    prediction = get_mask(unet, device, img_tensor)
    prediction = prediction.astype(int)
    prediction = np.expand_dims(prediction, axis=2)
    masked_image = original * prediction
    masked_image = masked_image.astype(np.uint8)
    tile = cv2.hconcat([original, masked_image])
    
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('tile.jpg', tile)
    cv2.imwrite('result.jpg', masked_image)