#include <iostream>
#include <string>
#include <vector>
#include <boost/python/numpy.hpp>
#include "Individual.hpp"

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
    Simulation(std::string mode, int num_preys, int num_predators,
	       int env_x, int env_y);

    void init_population();
    void clear_population();
    void reset_population();

    void set_mode(std::string mode);
    std::string get_mode() const;

    bp::list get_individuals_pos(std::vector<Individual> vector);
    bp::list get_preys_pos();
    bp::list get_predators_pos();

private:
    int env_x;
    int env_y;
    int num_preys;
    int num_predators;

    std::string mode;
    std::vector<Individual> preys;
    std::vector<Individual> predators;
    std::vector<int> test;

};
