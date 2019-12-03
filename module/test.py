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
confusion = False
timesteps = 2000

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

s = pylib.Simulation(input, output, num_preys, num_predators, env_x, env_y, eat_distance, confusion)

#run_simulation(s, 1)
#exit()

def smooth(x,window_len=11,window='hanning'):
    """
    smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

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

def eval_(pred_genotype,prey_genotype,confusion):

    s_ = pylib.Simulation(input, output, num_preys, num_predators, env_x, env_y, eat_distance, confusion)
    s_.reset_population()
    s_.load_prey_genotype(list(prey_genotype))
    s_.load_predator_genotype(list(pred_genotype))
    results = s_.run(timesteps)

    prey_fitness = results[2]
    pred_fitness = results[3]

    return -pred_fitness, -prey_fitness

def pred_eval(pred_indiv,preys_population,confusion):
    sum_pred_fitnesses = 0
    for prey_indiv in preys_population:
        pred_fitness, prey_fitness = eval_(pred_indiv,prey_indiv,confusion)
        sum_pred_fitnesses += pred_fitness
    return sum_pred_fitnesses / len(preys_population)


def prey_eval(preds_population,prey_indiv,confusion):
    sum_prey_fitnesses = 0
    for pred_indiv in preds_population:
        pred_fitness, prey_fitness = eval_(pred_indiv,prey_indiv,confusion)
        sum_prey_fitnesses += prey_fitness
    return sum_prey_fitnesses / len(preds_population)

def cmaes(nb_gen=15, confusion=True, display=True):

    es_preds = cma.CMAEvolutionStrategy(pred_genotype, 0.6)

    survivorships = []
    swarm_densitys = []
    swarm_dispersions = []

    pool = mp.Pool(mp.cpu_count())

    for i in range(nb_gen):
        preds_population = es_preds.ask()
        prey_genotype = np.random.rand(1, PREY_NETWORK_SIZE)

        pred_eval_part=partial(pred_eval, preys_population=prey_genotype, confusion=confusion)
        preds_fitnesses = pool.map(pred_eval_part, [pred_indiv for pred_indiv in preds_population])

        print("GENERATION {} : BEST = [{:1.2f}] AVERAGE = [{:1.2f}] WORST = [{:1.2f}]".format(i, -min(preds_fitnesses), -np.mean(preds_fitnesses), -max(preds_fitnesses)))

        es_preds.tell(preds_population, preds_fitnesses)

        pred = preds_population[np.argmin(preds_fitnesses)]

        results = []

        for prey in prey_genotype:
            s.reset_population()
            s.load_prey_genotype(list(prey))
            s.load_predator_genotype(list(pred))
            results.append(s.run(timesteps))

        results = np.mean(results, axis=0)

        density = results[0]
        dispersion = results[1]
        survivorship = results[4]

        survivorships.append(survivorship)

        #print("EVAL ON BEST : {} remaining preys".format(survivorships[-1]))

        swarm_densitys.append(density)
        swarm_dispersions.append(dispersion)

    return survivorships, swarm_densitys, swarm_dispersions

if __name__ == "__main__":

    t1 = time.time()
    survivorships, swarm_densitys, swarm_dispersions = cmaes(nb_gen=250,confusion=confusion)
    t2 = time.time()

    print("EVOLUTION LEARNING FINISHED IN : {} m {} s".format((t2 - t1) // 60, (t2 - t1) % 60))

    plt.figure()
    plt.title("survivorship")
    smoothed = smooth(survivorships)
    plt.plot(np.arange(len(smoothed)),smoothed,label="confusion")
    plt.legend()
    plt.show()
