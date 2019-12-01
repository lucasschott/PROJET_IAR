#include <iostream>
#include <string>
#include <vector>
#include <boost/python/numpy.hpp>
#include "Individual.hpp"
#include "Net.hpp"

namespace bp = boost::python;
namespace bn = boost::python::numpy;

#define X_POS 0
#define Y_POS 1

#define RAY_CASTING_MODE 0
#define GEOMETRIC_MODE 1

class Simulation
{

public:

    // Constructeur
    Simulation(int input_size, int output_size, int num_preys,
	       int num_predators, int env_x, int env_y,
	       double eat_distance, bool confusion);

    void init_population();
    void clear_population();
    void reset_population();
    void clear_population_observations();

    void load_prey_genotype(bp::list genotype);
    void load_predator_genotype(bp::list genotype);

    bp::list run(int timesteps);
    void compute_prey_observations();
    void compute_predator_observations();
    void eat_prey();
    std::vector<double> compute_swarm_density_and_dispersion();
    void apply_prey_actions(std::vector<int> &actions);
    void apply_predator_actions(std::vector<int> &actions);
    bool get_eat_flag(std::vector<int> observation);
    std::vector<int> forward_prey();
    std::vector<int> forward_predator();

    bp::list get_individuals_pos(std::vector<Individual> vector);
    bp::list get_preys_pos();
    bp::list get_predators_pos();

private:
    int env_x;
    int env_y;
    int num_preys;
    int num_predators;
    int input_size;
    int output_size;
    double eat_distance;
    bool confusion;

    std::vector<Individual> preys;
    std::vector<Individual> predators;

    Net prey_net;
    Net predator_net;
};
