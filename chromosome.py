import random
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm

class Chromosome:
    """
        Represents a chromosome in a genetic algorithm.

        Args:
            mut_prob (float): Mutation probability.
            recomb_prob (float): Recombination probability.
            epochs (int): Number of epochs for training.
            num_test (int): Number of tests to calculate fitness.
            dataloaders (dict): Dictionary of dataloaders for train and validation datasets.
            dataset_sizes (dict): Dictionary of dataset sizes for train and validation datasets.
        """
    def __init__(self, mut_prob, recomb_prob, epochs, num_test, dataloaders, dataset_sizes):
        self.net = {"extractor": 0, "mlp": []}
        self.mut_prob = mut_prob
        self.recomb_prob = recomb_prob
        self.fitness = 0
        self.epoches = epochs
        self.num_test = num_test
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_chromosome()

    def init_chromosome(self):
        """Initialize the chromosome with random values."""
        self.net["extractor"] = random.randint(1, 3)
        mlp = []
        for _ in range(random.randint(0, 2)):
            # (n_neurons, activation_function)
            hl = (random.choice([10, 20, 30]), random.randint(1, 2))
            mlp.append(hl)
        self.net["mlp"] = mlp.copy()

    def mut_ext(self):
        """Mutate the extractor gene."""
        prob = random.uniform(0, 1)
        if prob <= self.mut_prob:
            self.net["extractor"] = random.randint(1, 3)

    def mut_mlp(self):
        """Mutate the MLP genes."""
        for i in range(len(self.net['mlp'])):
            prob = random.uniform(0, 1)
            if prob <= self.mut_prob:
                h_new = (random.choice([10, 20, 30]), random.randint(1, 2))
                self.net['mlp'][i] = h_new

    def mut_pop(self):
        """Remove a gene from the MLP."""
        prob = random.uniform(0, 1)
        if prob <= self.mut_prob and len(self.net['mlp']) > 0:
            pop_id = random.randint(0, len(self.net['mlp']) - 1)
            self.net['mlp'].pop(pop_id)

    def mut_add(self):
        """Add a new gene to the MLP."""
        prob = random.uniform(0, 1)
        if prob <= self.mut_prob and len(self.net['mlp']) < 2:
            h_new = (random.choice([10, 20, 30]), random.randint(1, 2))
            app_id = random.randint(0, len(self.net['mlp']))
            self.net['mlp'].insert(app_id, h_new)

    def mutation(self):
        """Perform mutation on the chromosome."""
        self.mut_ext()
        self.mut_pop()
        self.mut_mlp()
        self.mut_add()
        self.calculate_fitness()

    def build_model(self):
        """
        Build the neural network model based on the chromosome.

        Returns:
            torch.nn.Module: Built neural network model.
        """
    # creating the model:
        # extractor == vgg11
        model = None
        num_features = 0
        if self.net['extractor'] == 1:
            model = models.vgg11(weights='DEFAULT')
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.classifier[0].in_features
            # classifier = list(model.classifier.children())[:-1]
            # model.classifier = nn.Sequential(*classifier)
        # resnet34 or resnet 18
        else:
            model = (models.resnet34(weights='DEFAULT')
                     if self.net['extractor'] == 2 else models.resnet18(weights='DEFAULT'))
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.fc.in_features
            # model.fc = nn.Identity()

        classifier = []
        prev_output = num_features
        for i in self.net['mlp']:
            classifier.append(nn.Linear(prev_output, i[0]))
            classifier.append(nn.ReLU(inplace=True) if i[1] == 1 else nn.Sigmoid())
            prev_output = i[0]

        classifier.append(nn.Linear(prev_output, 10))
        classifier.append(nn.Softmax(dim=1))

        if self.net['extractor'] == 1:
            model.classifier = nn.Sequential(*classifier)
        else:
            model.fc = nn.Sequential(*classifier)

        
        return model
        

    def calculate_fitness(self):
        """
            Calculate the fitness of the chromosome based on the built model and test performance.

            The fitness is calculated as the average accuracy over multiple tests.

            Returns:
                float: Fitness value.
            """
        

        for _ in range(self.num_test):
            
            model = self.build_model()
            model.to(self.device)
            # loss
            criterion = nn.CrossEntropyLoss()
            # optimizer
            optimizer = optim.Adam([{'params': model.classifier.parameters() if self.net['extractor'] == 1 else
                                model.fc.parameters(),'lr':0.001}], lr=0.001)
            #training
            model.train()
            with torch.set_grad_enabled(True):
                for epoch in range(self.epoches):
                    for x_train, y_train in tqdm(self.dataloaders['train'], desc=f"Epoch {epoch+1}", colour="blue"):
                        # Get the inputs and labels
                        x_train, y_train = x_train.to(self.device), y_train.to(self.device)

                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        # Forward + backward + optimize
                        outputs = model(x_train)
                        loss = criterion(outputs, y_train)
                        loss.backward()
                        optimizer.step()

            correct = 0
            total = 0
            model.eval()  
            with torch.set_grad_enabled(False):
                 for x_test, y_test in tqdm(self.dataloaders['val'], desc=f"Epoch {epoch+1}", colour="green"):
                    # Get the inputs and labels
                    x_test, y_test = x_test.to(self.device), y_test.to(self.device)

                    # Predict the classes of the inputs
                    outputs = model(x_test)
                    _, predicted = torch.max(outputs.data, 1)

                    # Update the number of correct predictions and total examples
                    total += y_test.size(0)
                    correct += (predicted == y_test).sum().item()
                    
            self.fitness += (correct/total)

        self.fitness = self.fitness / self.num_test





    
    



    