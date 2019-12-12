import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import shutil

import evol


def run(start=0,stop=10):

    for i in range(start,stop):

        evol.main(pop_size=10,nb_gen_pred=100,nb_gen=200,save_freq=20,
            conf_dir=conf_dir+"_{}".format(i),
            no_conf_dir=no_conf_dir+"_{}".format(i))


def plot(start=0,stop=10):

    conf_survivorships = []
    conf_swarm_densitys = []
    conf_swarm_dispersions = []
    no_conf_survivorships = []
    no_conf_swarm_densitys = []
    no_conf_swarm_dispersions = []

    for i in range(start,stop):
        conf_survivorships.append(np.load(conf_dir + "_{}".format(i) + "/survivorships.npy"))
        conf_swarm_densitys.append(np.load(conf_dir + "_{}".format(i) + "/swarm-densitys.npy"))
        conf_swarm_dispersions.append(np.load(conf_dir + "_{}".format(i) + "/swarm-dispersions.npy"))
        no_conf_survivorships.append(np.load(no_conf_dir + "_{}".format(i) + "/survivorships.npy"))
        no_conf_swarm_densitys.append(np.load(no_conf_dir + "_{}".format(i) + "/swarm-densitys.npy"))
        no_conf_swarm_dispersions.append(np.load(no_conf_dir + "_{}".format(i) + "/swarm-dispersions.npy"))

    conf_survivorships = np.array(conf_survivorships)
    conf_swarm_densitys = np.array(conf_swarm_densitys)
    conf_swarm_dispersions = np.array(conf_swarm_dispersions)
    no_conf_survivorships = np.array(no_conf_survivorships)
    no_conf_swarm_densitys = np.array(no_conf_swarm_densitys)
    no_conf_swarm_dispersions = np.array(no_conf_swarm_dispersions)

    conf_survivorships_mean = np.mean(conf_survivorships,axis=0)
    conf_swarm_densitys_mean = np.mean(conf_swarm_densitys,axis=0)
    conf_swarm_dispersions_mean = np.mean(conf_swarm_dispersions,axis=0)
    no_conf_survivorships_mean = np.mean(no_conf_survivorships,axis=0)
    no_conf_swarm_densitys_mean = np.mean(no_conf_swarm_densitys,axis=0)
    no_conf_swarm_dispersions_mean = np.mean(no_conf_swarm_dispersions,axis=0)

    conf_survivorships_errors = np.std(conf_survivorships,axis=0)
    conf_swarm_densitys_errors = np.std(conf_swarm_densitys,axis=0)
    conf_swarm_dispersions_errors = np.std(conf_swarm_dispersions,axis=0)
    no_conf_survivorships_errors = np.std(no_conf_survivorships,axis=0)
    no_conf_swarm_densitys_errors = np.std(no_conf_swarm_densitys,axis=0)
    no_conf_swarm_dispersions_errors = np.std(no_conf_swarm_dispersions,axis=0)

    plt.figure()
    plt.title("Survivorship")
    plt.plot(np.arange(len(conf_survivorships_mean,)),conf_survivorships_mean,label="confusion")
    plt.fill_between(np.arange(len(conf_survivorships_mean)),conf_survivorships_mean-conf_survivorships_errors,conf_survivorships_mean+conf_survivorships_errors,alpha=0.5)
    plt.plot(np.arange(len(no_conf_survivorships_mean)),no_conf_survivorships_mean,label="no confusion")
    plt.fill_between(np.arange(len(no_conf_survivorships_mean,)),no_conf_survivorships_mean-no_conf_survivorships_errors,no_conf_survivorships_mean+no_conf_survivorships_errors,alpha=0.5)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Swarm density")
    plt.plot(np.arange(len(conf_swarm_densitys_mean)),conf_swarm_densitys_mean,label="confusion")
    plt.fill_between(np.arange(len(conf_swarm_densitys_mean)),conf_swarm_densitys_mean-conf_swarm_densitys_errors,conf_swarm_densitys_mean+conf_swarm_densitys_errors,alpha=0.5)
    plt.plot(np.arange(len(no_conf_swarm_densitys_mean)),no_conf_swarm_densitys_mean,label="no confusion")
    plt.fill_between(np.arange(len(no_conf_swarm_densitys_mean)),no_conf_swarm_densitys_mean-no_conf_swarm_densitys_errors,no_conf_swarm_densitys_mean+no_conf_swarm_densitys_errors,alpha=0.5)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Swarm dispersion")
    plt.plot(np.arange(len(conf_swarm_dispersions_mean)),conf_swarm_dispersions_mean,label="confusion")
    plt.fill_between(np.arange(len(conf_swarm_dispersions_mean)),conf_swarm_dispersions_mean-conf_swarm_dispersions_errors,conf_swarm_dispersions_mean+conf_swarm_dispersions_errors,alpha=0.5)
    plt.plot(np.arange(len(no_conf_swarm_dispersions_mean)),no_conf_swarm_dispersions_mean,label="no confusion")
    plt.fill_between(np.arange(len(no_conf_swarm_dispersions_mean)),no_conf_swarm_dispersions_mean-no_conf_swarm_dispersions_errors,no_conf_swarm_dispersions_mean+no_conf_swarm_dispersions_errors,alpha=0.5)
    plt.legend()
    plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--results_dir', default='results', type=str)
    parser.add_argument('--confusion_dir', default='confusion', type=str)
    parser.add_argument('--no_confusion_dir', default='no_confusion', type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--stop', default=10, type=int)
    parser.add_argument('--no-run', dest='run', action='store_false')
    parser.set_defaults(run=True)
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=True)

    args = parser.parse_args()

    results_dir = args.results_dir
    conf_dir = results_dir + "/" + args.confusion_dir
    no_conf_dir = results_dir + "/" + args.no_confusion_dir
    start = args.start
    stop = args.stop
    do_run = args.run
    do_plot = args.plot

    if do_run:
        if os.path.isdir(results_dir):
            ans = input(results_dir + "directory already exists, do you want to overwrite it ?")
            if ans=="yes" or ans=="y":
                shutil.rmtree(results_dir,ignore_errors=True)
            else:
                exit()
        os.mkdir(results_dir)
        run(start=start, stop=stop)
    if do_plot:
        plot(start=start, stop=stop)
