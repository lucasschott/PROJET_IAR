#include <boost/python.hpp>
#include "Simulation.hpp"

using namespace boost::python;

BOOST_PYTHON_MODULE(pylib)
{
    class_< Simulation >("Simulation", init<std::string, int, int, int, int>())
    .def("init_population", &Simulation::init_population)
    .def("clear_population", &Simulation::clear_population)
    .def("reset_population", &Simulation::reset_population)
    .def("get_predators_pos", &Simulation::get_predators_pos)
    .def("get_preys_pos", &Simulation::get_preys_pos)
    .add_property("mode", &Simulation::get_mode, &Simulation::set_mode);

    class_< Individual >("Individual", init<std::string, int, int>())
    .add_property("pos_x", &Individual::get_pos_x, &Individual::set_pos_x)
    .add_property("pos_y", &Individual::get_pos_y, &Individual::set_pos_y);
}
