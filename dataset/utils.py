import cv2
import numpy as np
import torch


def preprocess_image(path):

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[...,None],[1, 1, 3])
    img = img.astype('float32')
    mx = np.max(img)
    if mx:
        img/=mx

    img = np.transpose(img, (2, 0, 1))
    img_ten = torch.tensor(img)
    return img_ten

def preprocess_mask(path):

    msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    msk = msk.astype('float32')
    msk/=255.0
    msk_ten = torch.tensor(msk)

    return msk_ten