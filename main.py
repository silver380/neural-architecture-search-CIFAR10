from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms
from evolutionary import EvolutionaryAlgorithm
import numpy as np
import sys

EPOCHS = 5
NUM_TEST = 5
N_ITER = 10
POPULATION_SIZE = 10
MUT_PROB = 0.9
RECOMB_PROB = 0.5

BATCH_SIZE = 32
IMAGE_RSIZE = 32

histories = []
transform = transforms.Compose([
    transforms.Resize((IMAGE_RSIZE, IMAGE_RSIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

image_datasets = {'train': train_data, 'val': test_data}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

if __name__ == "__main__":
    """ Main function for running the evolutionary algorithm."""
    ea = EvolutionaryAlgorithm(N_ITER, MUT_PROB, RECOMB_PROB, POPULATION_SIZE, EPOCHS, NUM_TEST, dataloaders, dataset_sizes)
    best_ans, history = ea.run()
    histories.append(history)
    histories = np.array(histories)   
    np.savetxt("histories.csv", histories,
                delimiter = ",")

    print(f"Test Accuracy of the best model is{best_ans.fitness}")

    original_stdout = sys.stdout
    with open('ans.txt', 'w') as f:
                    sys.stdout = f
                    bes_ext = "vgg11"
                    if best_ans.net['extractor'] == 2:
                            bes_ext = "resnet34"
                    elif best_ans.net['extractor'] == 3:
                            bes_ext = "resnet18"
                    print(f"extractor for the best model is {bes_ext}")
                    print(f"number of added hidden layers: {len(best_ans.net['mlp'])}")
                    if len(best_ans.net['mlp']) > 0:
                            for i in range(len(best_ans.net['mlp'])):
                                    print(f"layer {i+1} neurons: {best_ans.net['mlp'][i][0]}")
                                    act =  "ReLU" if best_ans.net['mlp'][i][1] == 1 else "Sigmoid"
                                    print(f"layer {i+1} activation function: {act}")
                                    
                    sys.stdout = original_stdout
