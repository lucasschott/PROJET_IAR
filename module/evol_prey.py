import numpy as np
import time
import cma
from functools import partial
import matplotlib.pyplot as plt

import multiprocessing as mp
from build import pylib

PRED_NETWORK_SIZE = 520
PREY_NETWORK_SIZE = PRED_NETWORK_SIZE + 156

input = 12
output = 4
num_preys = 50
num_predators = 1
env_x = 512
env_y = 512
eat_distance = 15
confusion = True
timesteps = 2000
POP_SIZE = 10

pred_genotype = list(np.random.rand(PRED_NETWORK_SIZE))
prey_genotype = list(np.random.rand(PREY_NETWORK_SIZE))

def time_init():
    t1 = time.time()
    sa = pylib.Simulation(input, output, num_preys, num_predators, env_x, env_y, eat_distance, confusion)
    t2 = time.time()
    print("init in {:1.2f} s".format(t2 - t1))
    exit()

def run_simulation(s, iter):
    results = []
    t1 = time.time()
    for i in range(iter):
        s.reset_population()
        s.load_prey_genotype(prey_genotype)
        s.load_predator_genotype(pred_genotype)
        results.append(s.run(timesteps))

    t2 = time.time()
    total = t2 - t1
    per_simulation = total / iter
    print(results)
    print(iter, " simulation ran in : finished in {:1.2f} s [{:1.0f} fps] [{:3.2f} s per simulation]".format(t2 - t1, (iter * timesteps) / (t2 - t1), per_simulation))
    return results

"""
    Simulation object
    run returns an array : [0] density
                           [1] dispersion
                           [2] prey_fitness
                           [3] pred_fitness
                           [4] survivorship
"""

DENSITY = 0
DISPERSION = 1
PREY_FITNESS = 2
PRED_FITNESS = 3
SURVIVORSHIP = 4

s = pylib.Simulation(input, output, num_preys, num_predators, env_x, env_y, eat_distance, confusion)


#run_simulation(s, 1)
#exit()

def save(data, path):
    np.save(path, data)

def smooth(x,window_len=11,window='hanning'):

    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def __eval__(pred_genotype,prey_genotype,confusion):
    s_ = pylib.Simulation(input, output, num_preys, num_predators, env_x, env_y, eat_distance, confusion)
    s_.reset_population()
    s_.load_prey_genotype(list(prey_genotype))
    s_.load_predator_genotype(list(pred_genotype))
    results = s_.run(timesteps)

    """ Sign switch on fitnesses is necessary for CMAES """
    results[PRED_FITNESS] = -1 * results[PRED_FITNESS]
    results[PREY_FITNESS] = -1 * results[PREY_FITNESS]

    return results

def pred_eval(pred_indiv,preys_population,confusion):
    sum_pred_fitnesses = 0
    for prey_indiv in preys_population:
        sum_pred_fitnesses += __eval__(pred_indiv,prey_indiv,confusion)[PRED_FITNESS]
    return sum_pred_fitnesses / len(preys_population)


def prey_eval(prey_indiv, preds_population, confusion):
    sum_prey_fitnesses = 0
    for pred_indiv in preds_population:
        sum_prey_fitnesses += __eval__(pred_indiv,prey_indiv,confusion)[PREY_FITNESS]
    return sum_prey_fitnesses / len(preds_population)

def cmaes(nb_gen=15, confusion=True, display=True):

    es_preys = cma.CMAEvolutionStrategy(prey_genotype, 0.6)

    survivorships = []
    swarm_densitys = []
    swarm_dispersions = []
    best_prey= None

    pool = mp.Pool(mp.cpu_count())
    pred_genotype = [list(np.load('best_preds/best_pred.npy'))]

    for i in range(nb_gen):
        preys_population = es_preys.ask()

        prey_eval_part=partial(prey_eval, preds_population=pred_genotype, confusion=confusion)
        preys_fitnesses = pool.map(prey_eval_part, [prey_indiv for prey_indiv in preys_population])

        print("GENERATION {} : BEST = [{:1.2f}] AVERAGE = [{:1.2f}] WORST = [{:1.2f}]".format(i, -min(preys_fitnesses), -np.mean(preys_fitnesses), -max(preys_fitnesses)))

        es_preys.tell(preys_population, preys_fitnesses)

        results = []
        best_prey = preys_population[np.argmin(preys_fitnesses)]

        for pred in pred_genotype:
            results.append(__eval__(pred, best_prey, confusion))

        results = np.mean(results, axis=0)

        density = results[DENSITY]
        dispersion = results[DISPERSION]
        survivorship = results[SURVIVORSHIP]

        survivorships.append(survivorship)
        swarm_densitys.append(density)
        swarm_dispersions.append(dispersion)

    return survivorships, swarm_densitys, swarm_dispersions, best_prey

if __name__ == "__main__":

    t1 = time.time()
    survivorships, swarm_densitys, swarm_dispersions, best_prey= cmaes(nb_gen=100,confusion=confusion)
    t2 = time.time()

    print("EVOLUTION LEARNING FINISHED IN : {} m {} s".format((t2 - t1) // 60, (t2 - t1) % 60))

    plt.figure()
    plt.title("survivorship")
    smoothed = smooth(survivorships)
    plt.plot(np.arange(len(smoothed)),smoothed,label="confusion")
    plt.legend()
    plt.savefig('prey_evol.png')
    #plt.show()

    save(best_prey, "best_prey")

