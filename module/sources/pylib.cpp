#include <boost/python.hpp>
#include "Simulation.hpp"
#include <torch/torch.h>
//#include "Net.hpp"

using namespace boost::python;


int main()
{
/*
  Net net = Net(10, 10);

  std::cout << "FC1 : " << net.fc1 << std::endl;
  std::cout << "FC2 : " << net.fc2 << std::endl;
  std::cout << "FC3 : " << net.fc3 << std::endl;
  std::cout << "FC4 : " << net.fc4 << std::endl;

  torch::Tensor w_fc1 = (*net.fc1).weight;
  torch::Tensor b_fc1 = (*net.fc1).bias;

  std::cout << w_fc1 << std::endl;
  std::cout << b_fc1 << std::endl;

  std::cout << w_fc1[0][0] << std::endl;

  w_fc1[0][0] = 10;

  std::cout << w_fc1[0][0] << std::endl;

  std::vector<float> array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  torch::Tensor test = torch::rand({w_fc1.size(0), w_fc1.size(1)});


  std::cout << test << std::endl;

  test[0] = torch::tensor(array);

  std::cout << test << std::endl;


  int size = 0;

  torch::Tensor w_fc1 = (*net.fc1).weight;
  size += w_fc1.size(0) * w_fc1.size(1);

  torch::Tensor b_fc1 = (*net.fc1).bias;
  size += b_fc1.size(0);


  torch::Tensor w_fc2 = (*net.fc2).weight;
  size += w_fc2.size(0) * w_fc2.size(1);

  torch::Tensor b_fc2 = (*net.fc2).bias;
  size += b_fc2.size(0);


  torch::Tensor w_fc3 = (*net.fc3).weight;
  size += w_fc3.size(0) * w_fc3.size(1);

  torch::Tensor b_fc3 = (*net.fc3).bias;
  size += b_fc3.size(0);


  torch::Tensor w_fc4 = (*net.fc4).weight;
  size += w_fc4.size(0) * w_fc4.size(1);

  torch::Tensor b_fc4 = (*net.fc4).bias;
  size += w_fc4.size(0);

  std::vector<float> params(size, 0);

  std::cout << w_fc4 << std::endl;

  net.load_from_vector(params);

  std::cout << w_fc4 << std::endl;

  return 0;
  */
}

BOOST_PYTHON_MODULE(pylib)
{
    class_< Simulation >("Simulation", init<int, int, int, int, int, int,
			 double, bool>())
    .def("init_population", &Simulation::init_population)
    .def("clear_population", &Simulation::clear_population)
    .def("reset_population", &Simulation::reset_population)
    .def("load_prey_genotype", &Simulation::load_prey_genotype)
    .def("run", &Simulation::run)
    .def("load_predator_genotype", &Simulation::load_predator_genotype)
    .def("get_predators_pos", &Simulation::get_predators_pos)
    .def("get_preys_pos", &Simulation::get_preys_pos);

    class_< Individual >("Individual", init<std::string, double, double, double>())
    .add_property("pos_x", &Individual::get_pos_x, &Individual::set_pos_x)
    .add_property("pos_y", &Individual::get_pos_y, &Individual::set_pos_y);
}
