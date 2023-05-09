import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet34, resnet18, vgg11
from sklearn.metrics import confusion_matrix, f1_score
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import util
epohcs = 10
n_iter = 10 #util.n_iter
batch_size = 32
transform = transforms.Compose([
    transforms.Resize((64, 64)), # Resize images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True )

#loading the test data
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False )