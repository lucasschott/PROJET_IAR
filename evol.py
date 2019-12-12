import numpy as np
import argparse
import time
import cma
from functools import partial
import matplotlib.pyplot as plt

import multiprocessing as mp
from module.build import pylib
import os
import shutil




NET_INPUT = 12
NET_OUTPUT = 4

LAYER = 12
PRED_NETWORK_SIZE = NET_INPUT*LAYER+LAYER + LAYER*NET_OUTPUT+NET_OUTPUT
PREY_NETWORK_SIZE = (NET_INPUT*2)*LAYER+LAYER + LAYER*NET_OUTPUT+NET_OUTPUT

DENSITY = 0
DISPERSION = 1
PREY_FITNESS = 2
PRED_FITNESS = 3
SURVIVORSHIP = 4




def __eval__(pred_genotype, prey_genotype, confusion, timesteps, num_preys, num_preds, env_x, env_y, eat_distance):
    s_ = pylib.Simulation(NET_INPUT, NET_OUTPUT, num_preys, num_preds, env_x, env_y, eat_distance, confusion)
    s_.reset_population()
    s_.load_prey_genotype(list(prey_genotype))
    s_.load_predator_genotype(list(pred_genotype))
    results = s_.run(timesteps)

    results[PRED_FITNESS] = -1 * results[PRED_FITNESS]
    results[PREY_FITNESS] = -1 * results[PREY_FITNESS]

    return results


def eval(indivs, confusion, timesteps, num_preys, num_preds, env_x, env_y, eat_distance):
    pred_indiv,prey_indiv = indivs
    return __eval__(pred_indiv, prey_indiv, confusion, timesteps, num_preys, num_preds, env_x, env_y, eat_distance)




########## predator evolution ##########

def pred_evol(pred_genotype, nb_gen=100, popsize=10, confusion=True, timesteps=2000, num_preys=50, num_preds=1, env_x=512, env_y=512, eat_distance=9):

    opts = cma.CMAOptions()
    opts['popsize'] = popsize

    es_preds = cma.CMAEvolutionStrategy(pred_genotype, 0.5, opts)

    best_pred = None
    best_pred_fit = 0

    pool = mp.Pool(mp.cpu_count())

    for i in range(nb_gen):
        preds_population = es_preds.ask()
        preys_population = np.random.rand(popsize, PREY_NETWORK_SIZE)

        eval_part = partial(eval, confusion=confusion, timesteps=timesteps, num_preys=int(max(1,(i/nb_gen)*num_preys)), num_preds=num_preds, env_x=env_x, env_y=env_y, eat_distance=eat_distance)
        args = []

        for pred_indiv in preds_population:
            for prey_indiv in preys_population:
                args.append((pred_indiv,prey_indiv))

        all_fitnesses = pool.map(eval_part, args)

        all_fitnesses = np.array(all_fitnesses).reshape(popsize,popsize,-1)
        preds_results = np.mean(all_fitnesses, axis=1)
        preds_fitnesses = preds_results[:, PRED_FITNESS]

        print("GENERATION {} : BEST = [{:1.2f}] AVERAGE = [{:1.2f}] WORST = [{:1.2f}] CURRENT_BEST = [{:1.2f}]".format(i, -min(preds_fitnesses), -np.mean(preds_fitnesses), -max(preds_fitnesses), -best_pred_fit))

        es_preds.tell(preds_population, preds_fitnesses)

        idx = np.argmin(preds_fitnesses)

        if preds_fitnesses[idx] * preds_fitnesses[idx] / np.mean(preds_fitnesses) < best_pred_fit:
            best_pred = preds_population[idx]
            best_pred_fit = preds_fitnesses[idx]

    return best_pred




########## co evolution ##########

def co_evol(pred_genotype, prey_genotype, nb_gen=200, save_freq=60, popsize=10, confusion=True, timesteps=2000, num_preys=50, num_preds=1, env_x=512, env_y=512, eat_distance=9):

    opts = cma.CMAOptions()
    opts['popsize'] = popsize

    es_preds = cma.CMAEvolutionStrategy(pred_genotype, 0.5, opts)
    es_preys = cma.CMAEvolutionStrategy(prey_genotype, 0.5, opts)

    survivorships = []
    survivorships_errors = []
    swarm_densitys = []
    swarm_densitys_errors = []
    swarm_dispersions = []
    swarm_dispersions_errors = []

    best_pred = None
    best_prey = None

    best_pred_fit = 0
    best_prey_fit = 0

    pool = mp.Pool(mp.cpu_count())

    for i in range(nb_gen):
        preds_population = es_preds.ask(popsize)
        preys_population = es_preys.ask(popsize)

        eval_part = partial(eval, confusion=confusion, timesteps=timesteps, num_preys=num_preys, num_preds=num_preds, env_x=env_x, env_y=env_y, eat_distance=eat_distance)
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

        if i % save_freq==0 :
            if confusion:
                np.save(conf_dir + "/best_pred_{}".format(i), best_pred)
                np.save(conf_dir + "/best_prey_{}".format(i), best_prey)
            else:
                np.save(no_conf_dir + "/best_pred_{}".format(i), best_pred)
                np.save(no_conf_dir + "/best_prey_{}".format(i), best_prey)

        gen_results = np.mean(all_fitnesses.reshape(popsize * popsize, -1), axis=0)
        gen_errors = np.std(all_fitnesses.reshape(popsize * popsize, -1), axis=0)

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

def main(num_preys = 50,
        num_predators = 1,
        env_x = 512,
        env_y = 512,
        eat_distance = 9,
        timesteps = 2000,
        pop_size = 10,
        nb_gen_pred = 100,
        nb_gen = 200,
        save_freq = 10,
        train_pred = True,
        pred_path = "",
        conf_dir = "result_confusion",
        no_conf_dir = "result_no_confusion"):


    if train_pred == False and pred_path == "":
        print("You must specify a predator genotype path if pretraining is disabled by providing : --pred=PATH")
        exit()

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

    if train_pred:
        print("\nPRE-EVOL PRED\n")

        pred_genotype = list(np.random.rand(PRED_NETWORK_SIZE))
        t1 = time.time()
        pred_genotype = pred_evol(pred_genotype, nb_gen=nb_gen_pred, popsize=pop_size, confusion=False)
        t2= time.time()

        print("PRE EVOLUTION PREDATOR WITH RANDOM PREYS\nLEARNING WITHOUT CONFUSION FINISHED IN : {} m {} s".format(
            (t2 - t1) // 60, (t2 - t1) % 60))

        np.save("best_pred.npy", pred_genotype)

    else:
        pred_genotype = np.load(args.pred)

    print("\nCO-EVOL NO CONFUSION\n")

    prey_genotype = list(np.random.rand(PREY_NETWORK_SIZE))
    t1 = time.time()
    (survivorships, survivorships_errors, swarm_densitys, swarm_densitys_errors,
    swarm_dispersions, swarm_dispersions_errors, best_pred,
    best_prey) = co_evol(pred_genotype, prey_genotype, nb_gen=nb_gen, save_freq=save_freq, popsize=pop_size, confusion=False)
    t2 = time.time()

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

    np.save(conf_dir + "/survivorships", survivorships)
    np.save(conf_dir + "/survivorships-errors", survivorships_errors)
    np.save(conf_dir + "/swarm-densitys", swarm_densitys)
    np.save(conf_dir + "/swarm-densitys-errors", swarm_densitys_errors)
    np.save(conf_dir + "/swarm-dispersions", swarm_dispersions)
    np.save(conf_dir + "/swarm-dispersions-errors", swarm_dispersions_errors)
    np.save(conf_dir + "/best_pred", best_pred)
    np.save(conf_dir + "/best_prey", best_prey)

    print("CO-EVOLUTION LEARNING WITH CONFUSION FINISHED IN : {} m {} s".format((t2 - t1) // 60, (t2 - t1) % 60))







if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--num_preys', default=50, type=int)
    parser.add_argument('--num_predators', default=1, type=int)
    parser.add_argument('--eat_distance', default=9, type=int)
    parser.add_argument('--train_pred', dest='train_pred', action='store_true')
    parser.add_argument('--no-train-pred', dest='train_pred', action='store_false')
    parser.set_defaults(train_pred=True)
    parser.add_argument('--timesteps', default=2000, type=int)
    parser.add_argument('--pred', default='', type=str)
    parser.add_argument('--env_x', default=512, type=int)
    parser.add_argument('--env_y', default=512, type=int)
    parser.add_argument('--popsize', default=5, type=int)
    parser.add_argument('--nb_gen_pred', default=50, type=int)
    parser.add_argument('--nb_gen', default=100, type=int)
    parser.add_argument('--save_freq', default=20, type=int)
    parser.add_argument('--conf_dir', default='result_confusion', type=str)
    parser.add_argument('--no_conf_dir', default='result_no_confusion', type=str)


    args = parser.parse_args()

    num_preys = args.num_preys
    num_predators = args.num_predators
    env_x = args.env_x
    env_y = args.env_y
    eat_distance = args.eat_distance
    timesteps = args.timesteps
    pop_size = args.popsize
    nb_gen_pred = args.nb_gen_pred
    nb_gen = args.nb_gen
    save_freq = args.save_freq
    train_pred = args.train_pred
    pred_path = args.pred
    conf_dir = args.conf_dir
    no_conf_dir = args.no_conf_dir

    
    main(num_preys = args.num_preys,
        num_predators = args.num_predators,
        env_x = args.env_x,
        env_y = args.env_y,
        eat_distance = args.eat_distance,
        timesteps = args.timesteps,
        pop_size = args.popsize,
        nb_gen_pred = args.nb_gen_pred,
        nb_gen = args.nb_gen,
        save_freq = args.save_freq,
        train_pred = args.train_pred,
        pred_path = args.pred,
        conf_dir = args.conf_dir,
        no_conf_dir = args.no_conf_dir)
