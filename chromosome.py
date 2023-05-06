import random

class Crhromosome:

    def __init__(self, mut_prob, recomb_prob, epochs):
        
        self.net = {"extractor":0,"mlp":[]}
        self.mut_prob = mut_prob
        self.recomb_prob =  recomb_prob
        self.fitness = 0
        self.epoches = epochs

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
            app_id = random.randint(0,2)
            self.net['mlp'].insert(app_id,h_new)



    