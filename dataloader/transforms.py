from torchvision import transforms
import numpy as np

norm = transforms.Normalize(
    np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
    np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
)

def square_resize_randomcrop(phase, image_size, pad_size):
    if phase == "train":
        t = [
            transforms.Resize([image_size+pad_size, image_size+pad_size]),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm
        ]
    else:
        t = [
            transforms.Resize([image_size+pad_size, image_size+pad_size]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            norm
        ]
    return transforms.Compose(t)

def reflectpad_randomcrop(phase, image_size, pad_size):
    if phase == "train":
        t = [
            transforms.RandomCrop(image_size,padding=pad_size,padding_mode='reflect'),
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

def ratio_resize_randomcrop(phase, image_size, pad_size):
    if phase == "train":
        t = [
            transforms.Resize(image_size+pad_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm
        ]
    else:
        t = [
            transforms.Resize(image_size+pad_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            norm
        ]
    return transforms.Compose(t)
