import numpy as np
import time
import cma
from functools import partial
import matplotlib.pyplot as plt

import multiprocessing as mp
from build import pylib
import os
import shutil


########## result directories ##########

conf_dir  = "result_confusion"
no_conf_dir  = "result_no_confusion"

if os.path.isdir(no_conf_dir):
    ans = input(no_conf_dir + " directory already exists, do you want to overwrite it ?")
    if ans=="yes" or ans=="y":
        shutil.rmtree(no_conf_dir, ignore_errors=True)
    else:
        exit()
if os.path.isdir(conf_dir):
    ans = input(conf_dir + " directory already exists, do you want to overwrite it ?")
    if ans=="yes" or ans=="y":
        shutil.rmtree(conf_dir, ignore_errors=True)
    else:
        exit()

os.mkdir(no_conf_dir)
os.mkdir(conf_dir)



########## evolution paramaters ##########

input = 12
output = 4

LAYER = 12
PRED_NETWORK_SIZE = input*LAYER+LAYER + 2*(LAYER*LAYER+LAYER) + LAYER*output+output
PREY_NETWORK_SIZE = (input*2)*LAYER+LAYER + 2*(LAYER*LAYER+LAYER) + LAYER*output+output

num_preys = 50
num_predators = 1
env_x = 512
env_y = 512
eat_distance = 9
timesteps = 100
pop_size = 2
nb_gen_pred = 1 #200
nb_gen = 1 #1200
save_freq = 1 #600



########## simulation/evaluation ##########

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


def eval(indivs,confusion):
    pred_indiv,prey_indiv = indivs
    return __eval__(pred_indiv,prey_indiv,confusion)



########## predator evolution ##########

def pred_evol(pred_genotype, nb_gen=100, popsize=20, confusion=True):
    
    opts = cma.CMAOptions()
    opts['popsize'] = popsize

    es_preds = cma.CMAEvolutionStrategy(pred_genotype, 0.6, opts)

    best_pred = None

    pool = mp.Pool(mp.cpu_count())

    for i in range(nb_gen):
        preds_population = es_preds.ask()
        prey_genotype = np.random.rand(PREY_NETWORK_SIZE)

        eval_part=partial(eval, confusion=confusion)
        args = []
        for pred_indiv in preds_population:
            args.append((pred_indiv,prey_genotype))

        all_fitnesses = pool.map(eval_part, args)

        all_fitnesses = np.array(all_fitnesses).reshape(popsize,1,-1)
        preds_results = np.mean(all_fitnesses, axis=1)
        preds_fitnesses = preds_results[:, PRED_FITNESS]

        print("GENERATION {} : BEST = [{:1.2f}] AVERAGE = [{:1.2f}] WORST = [{:1.2f}]".format(i, -min(preds_fitnesses), -np.mean(preds_fitnesses), -max(preds_fitnesses)))

        es_preds.tell(preds_population, preds_fitnesses)

        best_pred = preds_population[np.argmin(preds_fitnesses)]

    return best_pred



########## co evolution ##########

def co_evol(pred_genotype, prey_genotype, nb_gen=1200, save_freq=60, popsize=20, confusion=True):

    opts = cma.CMAOptions()
    opts['popsize'] = popsize

    es_preds = cma.CMAEvolutionStrategy(pred_genotype, 0.6, opts)
    es_preys = cma.CMAEvolutionStrategy(prey_genotype, 0.6, opts)

    survivorships = []
    survivorships_errors = []
    swarm_densitys = []
    swarm_densitys_errors = []
    swarm_dispersions = []
    swarm_dispersions_errors = []
    best_pred = None

    pool = mp.Pool(mp.cpu_count())

    for i in range(nb_gen):
        preds_population = es_preds.ask(popsize)
        preys_population = es_preys.ask(popsize)

        eval_part=partial(eval, confusion=confusion)
        args = []
        for pred_indiv in preds_population:
            for prey_indiv in preys_population:
                args.append((pred_indiv,prey_indiv))

        all_fitnesses = pool.map(eval_part, args)

        all_fitnesses = np.array(all_fitnesses).reshape(popsize,popsize,-1)
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

        if i%save_freq==0 :
            if confusion:
                np.save(conf_dir + "/best_pred_{}".format(i), best_pred)
                np.save(conf_dir + "/best_prey_{}".format(i), best_prey)
            else:
                np.save(no_conf_dir + "/best_pred_{}".format(i), best_pred)
                np.save(no_conf_dir + "/best_prey_{}".format(i), best_prey)

        gen_results = np.mean(preds_results, axis=0)
        gen_errors = np.std(preds_results, axis=0)

        density = gen_results[DENSITY]
        density_error = gen_errors[DENSITY]
        dispersion = gen_results[DISPERSION]
        dispersion_error = gen_errors[DISPERSION]
        survivorship = gen_results[SURVIVORSHIP]
        survivorship_error = gen_errors[SURVIVORSHIP]

        survivorships.append(survivorship)
        survivorships_errors.append(survivorship_error)
        swarm_densitys.append(density)
        swarm_densitys_errors.append(density_error)
        swarm_dispersions.append(dispersion)
        swarm_dispersions_errors.append(dispersion_error)

    return survivorships, survivorships_errors, swarm_densitys, swarm_densitys_errors, swarm_dispersions, swarm_dispersions_errors, best_pred, best_prey



########## main ##########

if __name__ == "__main__":



    print("\nPRE-EVOL PRED\n")

    pred_genotype = list(np.random.rand(PRED_NETWORK_SIZE))
    t1 = time.time()
    pred_genotype = pred_evol(pred_genotype, nb_gen=nb_gen_pred, popsize=pop_size, confusion=False)
    t2= time.time()

    print("PRE EVOLUTION PREDATOR WITH RANDOM PREYS\nLEARNING WITHOUT CONFUSION FINISHED IN : {} m {} s".format(
        (t2 - t1) // 60, (t2 - t1) % 60))



    print("\nCO-EVOL NO CONFUSION\n")

    prey_genotype = list(np.random.rand(PREY_NETWORK_SIZE))
    t1 = time.time()
    (survivorships, survivorships_errors, swarm_densitys, swarm_densitys_errors,
    swarm_dispersions, swarm_dispersions_errors, best_pred,
    best_prey) = co_evol(pred_genotype, prey_genotype, nb_gen=nb_gen, save_freq=save_freq, popsize=pop_size, confusion=False)
    t2 = time.time()

    print("Saving to : ", no_conf_dir)

    np.save(no_conf_dir + "/survivorships", survivorships)
    np.save(no_conf_dir + "/survivorships-errors", survivorships_errors)
    np.save(no_conf_dir + "/swarm-densitys", swarm_densitys)
    np.save(no_conf_dir + "/swarm-densitys-errors", swarm_densitys_errors)
    np.save(no_conf_dir + "/swarm-dispersions", swarm_dispersions)
    np.save(no_conf_dir + "/swarm-dispersions-errors", swarm_dispersions_errors)
    np.save(no_conf_dir + "/best_pred", best_pred)
    np.save(no_conf_dir + "/best_prey", best_prey)

    print("CO-EVOLUTION LEARNING WITHOUT CONFUSION FINISHED IN : {} m {} s".format((t2 - t1) // 60, (t2 - t1) % 60))



    print("\nCO-EVOL CONFUSION\n")

    prey_genotype = list(np.random.rand(PREY_NETWORK_SIZE))
    t1 = time.time()
    survivorships, survivorships_errors, swarm_densitys, swarm_densitys_errors, swarm_dispersions, swarm_dispersions_errors, best_pred, best_prey = co_evol(pred_genotype, prey_genotype, nb_gen=nb_gen, save_freq=save_freq, popsize=pop_size, confusion=True)
    t2 = time.time()

    print("Saving to ", conf_dir)
    np.save(conf_dir + "/survivorships", survivorships)
    np.save(conf_dir + "/survivorships-errors", survivorships_errors)
    np.save(conf_dir + "/swarm-densitys", swarm_densitys)
    np.save(conf_dir + "/swarm-densitys-errors", swarm_densitys_errors)
    np.save(conf_dir + "/swarm-dispersions", swarm_dispersions)
    np.save(conf_dir + "/swarm-dispersions-errors", swarm_dispersions_errors)
    np.save(conf_dir + "/best_pred", best_pred)
    np.save(conf_dir + "/best_prey", best_prey)

    print("CO-EVOLUTION LEARNING WITH CONFUSION FINISHED IN : {} m {} s".format((t2 - t1) // 60, (t2 - t1) % 60))

