import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--confusion_dir', default='result_confusion', type=str)
    parser.add_argument('--no_confusion_dir', default='result_no_confusion', type=str)

    args = parser.parse_args()

    conf_dir = args.confusion_dir
    no_conf_dir = args.no_confusion_dir

    conf_survivorships = np.load(conf_dir + "/survivorships-confusion.npy")
    conf_survivorships_errors = np.load(conf_dir + "/survivorships-errors-confusion.npy")
    conf_swarm_densitys = np.load(conf_dir + "/swarm-densitys-confusion.npy")
    conf_swarm_densitys_errors = np.load(conf_dir + "/swarm-densitys-errors-confusion.npy")
    conf_swarm_dispersions = np.load(conf_dir + "/swarm-dispersions-confusion.npy")
    conf_swarm_dispersions_errors = np.load(conf_dir + "/swarm-dispersions-errors-confusion.npy")
    conf_best_pred = np.load(conf_dir + "/best_pred_confusion.npy")
    conf_best_prey = np.load(conf_dir + "/best_prey_confusion.npy")

    no_conf_survivorships = np.load(no_conf_dir + "/survivorships-no-confusion.npy")
    no_conf_survivorships_errors = np.load(no_conf_dir + "/survivorships-errors-no-confusion.npy")
    no_conf_swarm_densitys = np.load(no_conf_dir + "/swarm-densitys-no-confusion.npy")
    no_conf_swarm_densitys_errors = np.load(no_conf_dir + "/swarm-densitys-errors-no-confusion.npy")
    no_conf_swarm_dispersions = np.load(no_conf_dir + "/swarm-dispersions-no-confusion.npy")
    no_conf_swarm_dispersions_errors = np.load(no_conf_dir + "/swarm-dispersions-errors-no-confusion.npy")
    no_conf_best_pred = np.load(no_conf_dir + "/best_pred_no_confusion.npy")
    no_conf_best_prey = np.load(no_conf_dir + "/best_prey_no_confusion.npy")

    plt.figure()
    plt.title("survivorship")
    plt.errorbar(np.arange(len(conf_survivorships)),conf_survivorships,yerr=conf_survivorships_errors,label="confusion")
    plt.errorbar(np.arange(len(no_conf_survivorships)),no_conf_survivorships,yerr=no_conf_survivorships_errors,label="no confusion")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("swarm density")
    plt.errorbar(np.arange(len(conf_swarm_densitys)),conf_swarm_densitys,yerr=conf_swarm_densitys_errors,label="confusion")
    plt.errorbar(np.arange(len(no_conf_swarm_densitys)),no_conf_swarm_densitys,yerr=no_conf_swarm_densitys_errors,label="no confusion")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("swarm dispersion")
    plt.errorbar(np.arange(len(conf_swarm_dispersions)),conf_swarm_dispersions,yerr=conf_swarm_dispersions_errors,label="confusion")
    plt.errorbar(np.arange(len(no_conf_swarm_dispersions)),no_conf_swarm_dispersions,yerr=no_conf_swarm_dispersions_errors,label="no confusion")
    plt.legend()
    plt.show()
