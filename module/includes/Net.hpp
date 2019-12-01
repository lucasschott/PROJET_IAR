#include <torch/torch.h>
#include <vector>

// Define a new Module.
class Net : torch::nn::Module
{

  public:

	  Net(int input, int output);
	  torch::Tensor forward(torch::Tensor x);
	  void load_from_vector(std::vector<float> params);

	  void load_linear_from_vector(torch::nn::Linear module,
				       std::vector<float> &params);

	template < typename T >
	std::vector<std::vector<T>> to_2d(const std::vector<T>& flat_vec,
					       std::size_t ncols )
	{
	    const auto nrows = flat_vec.size() / ncols ;
	    std::vector< std::vector<T> > mtx ;
	    const auto begin = std::begin(flat_vec) ;

	    for( std::size_t row = 0 ; row < nrows ; ++row )
	    {
		    mtx.push_back({begin + row*ncols, begin + (row+1)*ncols});
	    }
	    return mtx ;
	}

  torch::nn::Linear fc1 = nullptr;
  torch::nn::Linear fc2 = nullptr;
  torch::nn::Linear fc3 = nullptr;
  torch::nn::Linear fc4 = nullptr;
};
