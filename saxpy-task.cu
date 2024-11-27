#include <iterator>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/iterator/detail/zip_iterator.inl>
#include <thrust/transform.h>
#include <vector>
#include <random>
#include <cstdlib>
#include <charconv>
#include <format>
#include <ranges>
#include <concepts>


void usage(const char* filename)
{
	printf("Calculating a saxpy transform for two random vectors of the given length.\n");
	printf("Usage: %s <n>\n", filename);
}

// using namespace thrust;
//using namespace thrust::placeholders;

// TODO: Please refer to sorting examples:
// http://code.google.com/p/thrust/
// http://code.google.com/p/thrust/wiki/QuickStartGuide#Transformations
using el_type = float;

class saxpy_callable{
private:
	el_type a;

public:
	saxpy_callable(el_type a): a(a) {}

	__host__ __device__ el_type operator()(const el_type &x, el_type &y){
		return a * x + y;
	}
};

using dev_vec_it 		= thrust::device_vector<el_type>::iterator;
using vector_input_it	= std::vector<std::string_view>::iterator;

template<... Args>
std::tuple<Args...> validate_input_parameters(vector_input_it begin, vector_input_it end){
	bool is_valid{};
	
	// check count of input parameters
	is_valid &= std::distance(begin, end) == 2;

	// check is 1 argument can be interpreted as int > 0
	auto chars_start  = begin[1].data();
	auto chars_end 	= chars_start + begin[1].size();

	int converted{-1};
	std::from_chars(chars_start, chars_end, converted);
	
	is_valid &=  converted > 0;
	usage(begin->data());

	return std::make_tuple(is_valid, converted);
}

int main(int argc, char* argv[])
{
	constexpr size_t PRINTABLE_BUFF_SIZE = 128;
	std::vector<std::string_view> input_arguments(argv, argv + argc);

	bool is_valid;
	int n;
	std::tie(is_valid, n) = validate_input_parameters<bool, int>(input_arguments.begin(),  input_arguments.end());

	if (is_valid){
		usage(argv[0]);
		return 0;
	}

	cudaSetDevice(2);

	// TODO: Generate 3 vectors on host ( z = a * x + y)

	thrust::host_vector<el_type> host_x(n);
	thrust::host_vector<el_type> host_y(n);
	thrust::host_vector<el_type> host_z(n);

	{
		float rand_min(0), rand_max(999); 
		std::random_device rd;
    	std::mt19937 gen(rd());
    	std::uniform_real_distribution<el_type> dis(rand_min, rand_max); 

		thrust::generate(host_x.begin(), host_x.end(), [&]{return dis(gen);});
		thrust::generate(host_y.begin(), host_y.end(), [&]{return dis(gen);});
	}


	// Print out the input data if n is small.
	if (n <= PRINTABLE_BUFF_SIZE)
	{
		printf("Input data:\n");
		for(auto [el_x, el_y]: thrust::make_zip_iterator(thrust::make_tuple(host_x, host_y))){
			println(std::cout, "\t{} \t{}", el_x, el_y);
		}
	}

	// TODO: Transfer data to the device.
	thrust::device_vector<el_type> dev_x = host_x;
	thrust::device_vector<el_type> dev_y = host_x;
	thrust::device_vector<el_type> dev_z = host_z;
	const float a{2.5f};

	// TODO: Use transform to make an saxpy operation
	auto func = saxpy_callable(a);
	thrust::transform(dev_x.begin(), dev_x.end(), dev_y.begin(), dev_z.begin(), func);

	// TODO: Transfer data back to host.
	dev_z = host_z;

	// Print out the output data if n is small.
	if (n <= PRINTABLE_BUFF_SIZE)
	{
		std::cout << "OUTPUT DATA: \n";
		for(auto [el_x, el_y, el_z]: thrust::make_zip_iterator(thrust::make_tuple(host_x, host_y, host_z))){
			std::println(std::cout, "\t{} * {} + {} = {}", a, el_x, el_y, el_z);
		}
	}

	return 0;
}

