from torchvision import transforms
import numpy as np

norm = transforms.Normalize(
    np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
    np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
)

def resize_randomcrop(phase):
    if phase == "train":
        t = [
            transforms.Resize([92, 92]),
            transforms.RandomCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm
        ]
    else:
        t = [
            transforms.Resize([92, 92]),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            norm
        ]
    return transforms.Compose(t)

def reflectpad_randomcrop(phase):
    if phase == "train":
        t = [
            transforms.RandomCrop(84,padding=8,padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm
        ]
    else:
        t = [
            transforms.ToTensor(),
            norm
        ]
    return transforms.Compose(t)
