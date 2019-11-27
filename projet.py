import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cma
import simulation as sim
import collections
import multiprocessing as mp
from functools import partial


import torch
import torch.nn as nn
import torch.nn.functional as F




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cuda = False

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    cuda = True
    print("GPU MODE ENABLED")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    print("CPU MODE ENABLED")


class Net(nn.Module):

    def __init__(self,input,output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input,12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 12)
        self.fc4 = nn.Linear(12, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x




def eval(pred_indiv,prey_indiv,confusion):
    pred_fitness, prey_fitness = sim.simulation(pred_indiv,prey_indiv,confusion,False)
    return pred_fitness, prey_fitness

def pred_eval(pred_indiv,preys_population,confusion):
    sum_pred_fitnesses = 0
    for prey_indiv in preys_population:
        pred_fitness, prey_fitness = eval(pred_indiv,prey_indiv,confusion)
        sum_pred_fitnesses += pred_fitness
    return sum_pred_fitnesses


def prey_eval(preds_population,prey_indiv,confusion):
    sum_prey_fitnesses = 0
    for pred_indiv in preds_population:
        pred_fitness, prey_fitness = eval(pred_indiv,prey_indiv,confusion)
        sum_prey_fitnesses += prey_fitness
    return sum_prey_fitnesses



def dict_to_list(my_dict):
    my_list = []
    my_list.append(my_dict["fc1.weight"].tolist())
    my_list.append(my_dict["fc1.bias"].tolist())
    my_list.append(my_dict["fc2.weight"].tolist())
    my_list.append(my_dict["fc2.bias"].tolist())
    my_list.append(my_dict["fc3.weight"].tolist())
    my_list.append(my_dict["fc3.bias"].tolist())
    my_list.append(my_dict["fc4.weight"].tolist())
    my_list.append(my_dict["fc4.bias"].tolist())
    w1,b1,w2,b2,w3,b3,w4,b4 = my_list
    w1 = np.array(w1).flatten().tolist()
    w2 = np.array(w2).flatten().tolist()
    w3 = np.array(w3).flatten().tolist()
    w4 = np.array(w4).flatten().tolist()
    return w1 + b1 + w2 +b2 + w3 + b3 + w4 + b4

def list_to_dict(input,my_list):
    my_list = np.array(my_list)
    my_dict = collections.OrderedDict()
    a = 0
    b = input*12
    my_dict["fc1.weight"] = my_list[a:b]
    a = b
    b += 12
    my_dict["fc1.bias"] = my_list[a:b]
    a = b
    b += 12*12
    my_dict["fc2.weight"] = my_list[a:b]
    a = b
    b += 12
    my_dict["fc2.bias"] = my_list[a:b]
    a = b
    b += 12*12
    my_dict["fc3.weight"] = my_list[a:b]
    a = b
    b += 12
    my_dict["fc3.bias"] = my_list[a:b]
    a = b
    b += 12*4
    my_dict["fc4.weight"] = my_list[a:b]
    a = b
    b += 4
    my_dict["fc4.bias"] = my_list[a:b]
    return my_dict





def cmaes(nb_gen=100, confusion=True, display=True):

    
    pred_nn = Net(input=12,output=4)
    pred_genotype = pred_nn.state_dict()
    pred_genotype = dict_to_list(pred_genotype)

    prey_nn = Net(input=24,output=4)
    prey_genotype = prey_nn.state_dict()
    prey_genotype = dict_to_list(prey_genotype)

    es_preds = cma.CMAEvolutionStrategy(pred_genotype, 1)
    es_preys = cma.CMAEvolutionStrategy(prey_genotype, 1)

    survivorships = []
    swarm_densitys = []
    swarm_dispersions = []

    #while not es_preds.stop() and not es_preys.stop():
    for _ in range(nb_gen):
        preds_population = es_preds.ask()
        preys_population = es_preys.ask()

        pred_eval_part=partial(pred_eval, preys_population=preys_population, confusion=confusion)
        prey_eval_part=partial(prey_eval, preds_population=preds_population, confusion=confusion)

        pool = mp.Pool(1)#mp.cpu_count())
        preds_fitnesses = pool.map(pred_eval_part, [pred_indiv for pred_indiv in preds_population])
        pool = mp.Pool(mp.cpu_count())
        preys_fitnesses = pool.map(prey_eval_part, [prey_indiv for prey_indiv in preys_population])

        es_preds.tell(preds_population, preds_fitnesses)
        es_preys.tell(preys_population, preys_fitnesses)
        
        pred = es_preds.ask(number=1)[0]
        prey = es_preys.ask(number=1)[0]

        pred_fitness, prey_fitness, survivorship, swarm_density, swarm_dispersion = sim.simulation(pred,prey,confusion,True)
        survivorships.append(survivorship)
        swarm_densitys.append(swarms_density)
        swarm_dispersions.append(swarm_dispersion)

    return survivorships, swarm_densitys, swarm_dispersions




if __name__ == "__main__":
    survivorships, swarm_densitys, swarm_dispersions = cmaes(nb_gen=50,confusion=False)
    survivorships_confusion, swarm_densitys_confusion, swarm_dispersions_confusion = cmaes(nb_gen=50,confusion=True)
    
    plt.figure()
    plt.title("survivorship")
    plt.plot(np.arange(len(survivorships_confusion)),survivorship_confusion,label="confusion")
    plt.plot(np.arange(len(survivorships)),survivorship,label="no confusion")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("swarm density")
    plt.plot(np.arange(len(swarm_densitys_confusion)),swarm_densitys_confusion,label="confusion")
    plt.plot(np.arange(len(swarm_densitys)),swarm_densitys,label="no confusion")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("swarm dispersion")
    plt.plot(np.arange(len(swarm_dispersions_confusion)),swarm_dispersions_confusion,label="confusion")
    plt.plot(np.arange(len(swarm_dispersions)),swarm_dispersions,label="no confusion")
    plt.legend()
    plt.show()
