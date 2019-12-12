#include "Net.hpp"

Net::Net(int input, int output)
{
	    fc1 = register_module("fc1", torch::nn::Linear(input, 12));
	    fc2 = register_module("fc2", torch::nn::Linear(12, output));
}

torch::Tensor Net::forward(torch::Tensor x)
{
	    x = torch::sigmoid(fc1->forward(x));
	    x = torch::sigmoid(fc2->forward(x));

	    return x;
}

std::vector<float> get_matching_data(torch::Tensor tensor,
				     std::vector<float> params)
{
	std::vector<float>::const_iterator first = params.begin();
	int last_index = 0;

	if (tensor.dim() == 2)
		last_index += tensor.size(0) * tensor.size(1);
	else
		last_index += tensor.size(0);

	std::vector<float>::const_iterator last = params.begin() + last_index;

	return std::vector<float>(first, last);
}

void Net::load_linear_from_vector(torch::nn::Linear module,
				  std::vector<float> &params)
{

	torch::Tensor tensor_weight = (*module).weight;
	torch::Tensor tensor_bias = (*module).bias;

	auto weights = to_2d(get_matching_data(tensor_weight, params),
			     tensor_weight.size(1));

	auto bias = get_matching_data(tensor_bias, params);

	int used = tensor_weight.size(0) * tensor_weight.size(1) +
		tensor_bias.size(0);


	for (std::vector<std::vector<float> >::iterator it = weights.begin() ;
	     it != weights.end(); ++it)
	{
		auto index = std::distance(weights.begin(), it);
		tensor_weight[index] = torch::tensor(*it);
	}

	tensor_bias = torch::tensor(bias);

	params.erase(params.begin(), params.begin() + used);
}

void Net::load_from_vector(std::vector<float> params)
{
	load_linear_from_vector(this->fc1, params);
	load_linear_from_vector(this->fc2, params);
}
