import random
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

class Chromosome:

    def __init__(self, mut_prob, recomb_prob, epochs, num_test, train_dataloader, test_dataloader):
        
        self.net = {"extractor":0,"mlp":[]}
        self.mut_prob = mut_prob
        self.recomb_prob =  recomb_prob
        self.fitness = 0
        self.epoches = epochs
        self.num_test = num_test
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_chromosome(self):
        self.net["extractor"] = random.randint(1,3)
        mlp = []
        for _ in range(random.randint(0,3)):
            # (n_neurons, activation_function)
            hl = (random.choice([10,20,30]),random.randint(1,2))
            mlp.append(hl)
        self.net["mlp"] = mlp.copy()
    
    def mut_ext(self):
        prob = random.uniform(0,1)
        if prob <= self.mut_prob:
            self.net["extractor"] = random.randint(1,3)
    
    def mut_mlp(self):
        for i in range(len(self.net['mlp'])):
            prob = random.uniform(0,1)
            if prob <= self.mut_prob:
                h_new = (random.choice([10,20,30]),random.randint(1,2))
                self.net['mlp'][i] = h_new
    
    def mut_pop(self):
        prob = random.uniform(0,1)
        if prob <= self.mut_prob and len(self.net['mlp']) > 0:
            pop_id = random.randint(len(self.net['mlp'])-1)
            self.net['mlp'].pop(pop_id)

    def mut_add(self):
        prob = random.uniform(0,1)
        if prob <= self.mut_prob and len(self.net['mlp']) < 3:
            h_new = (random.choice([10,20,30]),random.randint(1,2))
            app_id = random.randint(0,len(self.net['mlp']))
            self.net['mlp'].insert(app_id,h_new)

    def calculate_fitness(self):

        # creating the model:
        # extractor == vgg11
        model = None
        num_features = 0
        if self.net['extractor']==0:
            model = models.vgg11(pretraind=True)
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
            classifier.append(nn.Linear(prev_output,i[0]))
            classifier.append(nn.ReLU(inplace=True) if i[1]==1 else nn.Sigmoid())
        
        classifier.append(nn.Softmax(dim=1))

        if self.net['extractor'] == 0:
            model.classifier = nn.Sequential(*classifier)
        else:
            model.fc = nn.Sequential(*classifier)
        model.to(self.device)

        #loss
        criterion = nn.CrossEntropyLoss()
        #optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        #training and testing the model:
        test_acc = 0
        for _ in range(self.num_test):
            #training
            for _ in range(self.epoches):
                for x_train, y_train in self.train_dataloader:
                    # Get the inputs and labels
                    x_train, y_train = x_train.to(self.device), y_train.to(self.sdevice)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward + backward + optimize
                    outputs = model(x_train)
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()

            correct = 0
            total = 0
            with torch.no_grad():
                for x_test, y_test in self.testloader:
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








    
    



    