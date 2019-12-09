import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--confusion_dir', default='result_confusion', type=str)
    parser.add_argument('--no_confusion_dir', default='result_no_confusion', type=str)

    args = parser.parse_args()

    conf_dir = args.confusion_dir
    no_conf_dir = args.no_confusion_dir

    conf_survivorships = np.load(conf_dir + "/survivorships.npy")
    conf_survivorships_errors = np.load(conf_dir + "/survivorships-errors.npy")
    conf_swarm_densitys = np.load(conf_dir + "/swarm-densitys.npy")
    conf_swarm_densitys_errors = np.load(conf_dir + "/swarm-densitys-errors.npy")
    conf_swarm_dispersions = np.load(conf_dir + "/swarm-dispersions.npy")
    conf_swarm_dispersions_errors = np.load(conf_dir + "/swarm-dispersions-errors.npy")
    conf_best_pred = np.load(conf_dir + "/best_pred.npy")
    conf_best_prey = np.load(conf_dir + "/best_prey.npy")

    no_conf_survivorships = np.load(no_conf_dir + "/survivorships.npy")
    no_conf_survivorships_errors = np.load(no_conf_dir + "/survivorships-errors.npy")
    no_conf_swarm_densitys = np.load(no_conf_dir + "/swarm-densitys.npy")
    no_conf_swarm_densitys_errors = np.load(no_conf_dir + "/swarm-densitys-errors.npy")
    no_conf_swarm_dispersions = np.load(no_conf_dir + "/swarm-dispersions.npy")
    no_conf_swarm_dispersions_errors = np.load(no_conf_dir + "/swarm-dispersions-errors.npy")
    no_conf_best_pred = np.load(no_conf_dir + "/best_pred.npy")
    no_conf_best_prey = np.load(no_conf_dir + "/best_prey.npy")

    plt.figure()
    plt.title("survivorship")
    plt.plot(np.arange(len(conf_survivorships)),conf_survivorships,label="confusion")
    plt.fill_between(np.arange(len(conf_survivorships)),conf_survivorships-conf_survivorships_errors,conf_survivorships+conf_survivorships_errors,alpha=0.5)
    plt.plot(np.arange(len(no_conf_survivorships)),no_conf_survivorships,label="no confusion")
    plt.fill_between(np.arange(len(no_conf_survivorships)),no_conf_survivorships-no_conf_survivorships_errors,no_conf_survivorships+no_conf_survivorships_errors,alpha=0.5)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("swarm density")
    plt.plot(np.arange(len(conf_swarm_densitys)),conf_swarm_densitys,label="confusion")
    plt.fill_between(np.arange(len(conf_swarm_densitys)),conf_swarm_densitys-conf_swarm_densitys_errors,conf_swarm_densitys+conf_swarm_densitys_errors,alpha=0.5)
    plt.plot(np.arange(len(no_conf_swarm_densitys)),no_conf_swarm_densitys,label="no confusion")
    plt.fill_between(np.arange(len(no_conf_swarm_densitys)),no_conf_swarm_densitys-no_conf_swarm_densitys_errors,no_conf_swarm_densitys+no_conf_swarm_densitys_errors,alpha=0.5)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("swarm dispersion")
    plt.plot(np.arange(len(conf_swarm_dispersions)),conf_swarm_dispersions,label="confusion")
    plt.fill_between(np.arange(len(conf_swarm_dispersions)),conf_swarm_dispersions-conf_swarm_dispersions_errors,conf_swarm_dispersions+conf_swarm_dispersions_errors,alpha=0.5)
    plt.plot(np.arange(len(no_conf_swarm_dispersions)),no_conf_swarm_dispersions,label="no confusion")
    plt.fill_between(np.arange(len(no_conf_swarm_dispersions)),no_conf_swarm_dispersions-no_conf_swarm_dispersions_errors,no_conf_swarm_dispersions+no_conf_swarm_dispersions_errors,alpha=0.5)
    plt.legend()
    plt.show()
