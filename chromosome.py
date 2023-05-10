import random
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from train import train_model


class Chromosome:
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
        self.net["extractor"] = random.randint(1, 3)
        mlp = []
        for _ in range(random.randint(0, 2)):
            # (n_neurons, activation_function)
            hl = (random.choice([10, 20, 30]), random.randint(1, 2))
            mlp.append(hl)
        self.net["mlp"] = mlp.copy()

    def mut_ext(self):
        prob = random.uniform(0, 1)
        if prob <= self.mut_prob:
            self.net["extractor"] = random.randint(1, 3)

    def mut_mlp(self):
        for i in range(len(self.net['mlp'])):
            prob = random.uniform(0, 1)
            if prob <= self.mut_prob:
                h_new = (random.choice([10, 20, 30]), random.randint(1, 2))
                self.net['mlp'][i] = h_new

    def mut_pop(self):
        prob = random.uniform(0, 1)
        if prob <= self.mut_prob and len(self.net['mlp']) > 0:
            pop_id = random.randint(0, len(self.net['mlp']) - 1)
            self.net['mlp'].pop(pop_id)

    def mut_add(self):
        prob = random.uniform(0, 1)
        if prob <= self.mut_prob and len(self.net['mlp']) < 2:
            h_new = (random.choice([10, 20, 30]), random.randint(1, 2))
            app_id = random.randint(0, len(self.net['mlp']))
            self.net['mlp'].insert(app_id, h_new)

    def mutation(self):
        self.mut_ext()
        self.mut_pop()
        self.mut_mlp()
        self.mut_add()
        self.calculate_fitness()

    def calculate_fitness(self):
        # creating the model:
        # extractor == vgg11
        model = None
        num_features = 0
        if self.net['extractor'] == 0:
            model = models.vgg11(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.classifier[6].in_features
            classifier = list(model.classifier.children())[:-1]
            model.classifier = nn.Sequential(*classifier)
        # resnet34 or resnet 18
        else:
            model = (models.resnet34(pretrained=True)
                     if self.net['extractor'] == 1 else models.resnet18(pretrained=True))
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.fc.in_features
            model.fc = nn.Identity()

        classifier = []
        prev_output = num_features
        for i in self.net['mlp']:
            classifier.append(nn.Linear(prev_output, i[0]))
            classifier.append(nn.ReLU(inplace=True) if i[1] == 1 else nn.Sigmoid())
            prev_output = i[0]

        classifier.append(nn.Linear(prev_output, 10))
        classifier.append(nn.Softmax(dim=1))

        if self.net['extractor'] == 0:
            model.classifier = nn.Sequential(*classifier)
        else:
            model.fc = nn.Sequential(*classifier)
        model.to(self.device)

        # loss
        criterion = nn.CrossEntropyLoss()
        # optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        acc = train_model(model, criterion, optimizer, self.device, self.dataloaders, self.dataset_sizes, 5)
        #         #training and testing the model:
        #         test_acc = 0
        #         for _ in range(self.num_test):
        #             #training
        #             for epoch in range(self.epoches):
        #                 for x_train, y_train in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", colour="blue"):
        #                     # Get the inputs and labels
        #                     x_train, y_train = x_train.to(self.device), y_train.to(self.device)

        #                     # Zero the parameter gradients
        #                     optimizer.zero_grad()

        #                     # Forward + backward + optimize
        #                     outputs = model(x_train)
        #                     loss = criterion(outputs, y_train)
        #                     loss.backward()
        #                     optimizer.step()

        #             correct = 0
        #             total = 0
        #             with torch.no_grad():
        #                 for x_test, y_test in self.testloader:
        #                     # Get the inputs and labels
        #                     x_test, y_test = x_test.to(self.device), y_test.to(self.device)

        #                     # Predict the classes of the inputs
        #                     outputs = model(x_test)
        #                     _, predicted = torch.max(outputs.data, 1)

        #                     # Update the number of correct predictions and total examples
        #                     total += y_test.size(0)
        #                     correct += (predicted == y_test).sum().item()
        #             self.fitness += (correct/total)

        self.fitness = acc





    
    



    