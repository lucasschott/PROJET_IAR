#include <boost/python.hpp>
#include "Simulation.hpp"
#include <torch/torch.h>
//#include "Net.hpp"

using namespace boost::python;

BOOST_PYTHON_MODULE(pylib)
{
    class_< Simulation >("Simulation", init<int, int, int, int, int, int,
			 double, bool>())
    .def("init_population", &Simulation::init_population)
    .def("clear_population", &Simulation::clear_population)
    .def("reset_population", &Simulation::reset_population)
    .def("load_prey_genotype", &Simulation::load_prey_genotype)
    .def("run", &Simulation::run)
    .def("step", &Simulation::step)
    .def("load_predator_genotype", &Simulation::load_predator_genotype)
    .def("get_predators_pos", &Simulation::get_predators_pos)
    .def("get_preys_pos", &Simulation::get_preys_pos);
}
