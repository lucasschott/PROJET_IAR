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

""" PREDATORS * PREYS """

"""
    mean axis 0 : PREDATORS FITNESSES
    mean axis 1 : PREYS FITNESSES
"""

def pred_eval(pred_indiv,preys_population,confusion):
    results = []
    for prey_indiv in preys_population:
        results.append(__eval__(pred_indiv,prey_indiv,confusion))

    return results

def cmaes(nb_gen=15, popsize=10, confusion=True, display=True):

    opts = cma.CMAOptions()
    opts['popsize'] = popsize

    es_preds = cma.CMAEvolutionStrategy(pred_genotype, 0.6, opts)
    es_preys = cma.CMAEvolutionStrategy(prey_genotype, 0.6, opts)

    survivorships = []
    swarm_densitys = []
    swarm_dispersions = []
    best_pred = None

    pool = mp.Pool(mp.cpu_count())

    for i in range(nb_gen):
        preds_population = es_preds.ask(popsize)
        preys_population = es_preys.ask(popsize)

        pred_eval_part=partial(pred_eval, preys_population=preys_population, confusion=confusion)
        all_fitnesses = pool.map(pred_eval_part, [pred_indiv for pred_indiv in preds_population])

        all_fitnesses = np.array(all_fitnesses)
        preds_results = np.mean(all_fitnesses, axis=1)
        preys_results = np.mean(all_fitnesses, axis=0)

        preds_fitnesses = preds_results[:, PRED_FITNESS]
        preys_fitnesses = preys_results[:, PREY_FITNESS]

        print("GENERATION {}".format(i))
        print("PREDATORS :  BEST = [{:1.2f}] AVERAGE = [{:1.2f}] WORST = [{:1.2f}]".format(-min(preds_fitnesses), -np.mean(preds_fitnesses), -max(preds_fitnesses)))
        print("PREYS : BEST = [{:1.2f}] AVERAGE = [{:1.2f}] WORST = [{:1.2f}]".format(-min(preys_fitnesses), -np.mean(preys_fitnesses), -max(preys_fitnesses)))

        es_preds.tell(preds_population, preds_fitnesses)
        es_preys.tell(preys_population, preys_fitnesses)

        best_pred_idx = np.argmin(preds_fitnesses)
        best_prey_idx = np.argmin(preys_fitnesses)

        best_pred = preds_population[best_pred_idx]
        best_prey = preys_population[best_prey_idx]

        gen_results = np.mean(preds_results, axis=1)

        density = gen_results[DENSITY]
        dispersion = gen_results[DISPERSION]
        survivorship = gen_results[SURVIVORSHIP]

        survivorships.append(survivorship)
        swarm_densitys.append(density)
        swarm_dispersions.append(dispersion)

    return survivorships, swarm_densitys, swarm_dispersions, best_pred, best_prey

if __name__ == "__main__":

    t1 = time.time()
    survivorships, swarm_densitys, swarm_dispersions, best_pred, best_prey = cmaes(nb_gen=600, popsize=10, confusion=False)
    t2 = time.time()

    save(survivorships, "survivorships-no-confusion")
    save(swarm_densitys, "swarm-densitys-no-confusion")
    save(swarm_dispersions, "swarm-dispersions-no-confusion")
    save(best_pred, "best_pred_no_confusion")
    save(best_prey, "best_prey_no_confusion")

    print("EVOLUTION LEARNING WITHOUT CONFUSION FINISHED IN : {} m {} s".format((t2 - t1) // 60, (t2 - t1) % 60))

    t1 = time.time()
    survivorships, swarm_densitys, swarm_dispersions, best_pred, best_prey = cmaes(nb_gen=600, popsize=10, confusion=True)
    t2 = time.time()

    save(survivorships, "survivorships-confusion")
    save(swarm_densitys, "swarm-densitys-confusion")
    save(swarm_dispersions, "swarm-dispersions-confusion")
    save(best_pred, "best_pred_confusion")
    save(best_prey, "best_prey_confusion")

    print("EVOLUTION LEARNING WITH CONFUSION FINISHED IN : {} m {} s".format((t2 - t1) // 60, (t2 - t1) % 60))

