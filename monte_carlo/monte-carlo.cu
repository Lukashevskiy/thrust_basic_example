#include <cstdio> 
#include <cstdlib>

#include <cuda.h>
#include <curand.h>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>


using dtype = float;

void GPU_fill_rand(thrust::device_vector<dtype> &device_data_ptr, int data_nr_rows, int data_nr_cols, int seed=0) {
    curandGenerator_t generator_ptr;

	curandStatus_t status;
 
    status = curandCreateGenerator(&generator_ptr, CURAND_RNG_PSEUDO_DEFAULT);

    status = curandSetPseudoRandomGeneratorSeed(generator_ptr, seed);

    status = curandGenerateUniform(
						generator_ptr, 
						device_data_ptr.data().get(), 
						data_nr_rows * data_nr_cols);


}



struct tr {
	__host__ __device__
	dtype operator() (dtype& X) const {
		return X * X / (X + 1.f) + 1.f / X;
	}
};

struct to_range {
	// mode - режим (1 для преобразования Х)
	int mode;
	to_range(int _mode) : mode(_mode) {}

	__host__ __device__
	dtype operator() (dtype& vec) const {
		dtype a = 0.5f;
		dtype b = 5.f;
		if (mode == 1) {
			return a + (b - a) * vec;
		} else {
			return (b * b / (b + 1.f) + 1.f / b) * vec;
		}
	}
};

struct count_fx {
	__host__ __device__
	int operator() (dtype& fX, dtype &Y) const {
		if (fX < Y) { 
			return 0;
		} else {
			return 1;
		}
	}
};

int main(int argc, char* argv[]) 
{

	cudaSetDevice(2);

	dtype a = std::stod(argv[1]);
	dtype b = std::stod(argv[2]);

	for(int n: {100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 800'000'000}){
		auto time_begin = std::chrono::high_resolution_clock().now();

		thrust::device_vector<dtype> data_x_device(n), data_y_device(n);	

		thrust::device_vector<dtype> fX(n);
		thrust::device_vector<int> cnt(n);


		GPU_fill_rand(data_x_device, n, 1);
		GPU_fill_rand(data_y_device, n, 1);

		
		thrust::transform(data_x_device.begin(), data_x_device.end(), data_x_device.begin(), to_range(1));
		thrust::transform(data_y_device.begin(), data_y_device.end(), data_y_device.begin(), to_range(0));
		
		thrust::transform(data_x_device.begin(), data_x_device.end(), fX.begin(), tr());	



		thrust::transform(fX.begin(), fX.end(), data_y_device.begin(),  cnt.begin(), count_fx());	
		
		int result = thrust::reduce(cnt.begin(), cnt.end(), 0, thrust::plus<int>());
		
		tr functor;
		float area = result * (b - a) * functor(b) / (n) / 2;
		auto time_end = std::chrono::high_resolution_clock().now();
		auto duration_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_begin);
		
		std::cout << "---------------------------------------\n";
        std::cout << "Num samples \t- " <<  n << "\n";
		std::cout << "Trapesoid \t- " << area << "\n";
        std::cout << "TIME DURATION \t- " << duration_time.count() << " milliseconds\n";
        std::cout << "---------------------------------------\n";
		
	}
}
