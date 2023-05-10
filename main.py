from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms
from evolutionary import EvolutionaryAlgorithm

EPOCHS = 5
NUM_TEST = 5
N_ITER = 10
POPULATION_SIZE = 10
MUT_PROB = 0.1
RECOMB_PROB = 0.9

TRAIN_DATA_BATCH_SIZE = 4
TEST_DATA_BATCH_SIZE = 4
BATCH_SIZE = 100

# transform = transforms.Compose([
#     transforms.Resize((64, 64)), # Resize images
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                           std=[0.229, 0.224, 0.225])])

# train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
# train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True )

# #loading the test data
# test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)
# test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False )
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

image_datasets = {'train': train_data, 'val': test_data}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

ea = EvolutionaryAlgorithm(N_ITER, MUT_PROB, RECOMB_PROB, POPULATION_SIZE, EPOCHS, NUM_TEST, dataloaders, dataset_sizes)
fitness, history = ea.run()
print(history)
