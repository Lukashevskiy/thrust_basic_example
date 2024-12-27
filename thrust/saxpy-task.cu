#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <cstdio>
#include <cstdlib>

void usage(const char *filename)
{
	printf("Calculating a saxpy transform for two random vectors of the given length.\n");
	printf("Usage: %s <n>\n", filename);
}

// using namespace thrust::placeholders;

// TODO: Please refer to sorting examples:
// http://code.google.com/p/thrust/
// http://code.google.com/p/thrust/wiki/QuickStartGuide#Transformations

struct saxpy
{
	float a;
	// Constructor:
	saxpy(float a) : a(a) {}

	__host__ __device__ float operator()(float x, float y)
	{
		return a * x + y;
	}
};

int main(int argc, char *argv[])
{
	const int printable_n = 128;

	if (argc != 2)
	{
		usage(argv[0]);
		return 0;
	}

	int n = atoi(argv[1]);
	if (n <= 0)
	{
		usage(argv[0]);
		return 0;
	}
	cudaSetDevice(2);
	
	thrust::default_random_engine rngX(24);
	thrust::default_random_engine rngY(12);
	thrust::uniform_real_distribution<float> dist(-1000.0, 1000.0);
	
	thrust::host_vector<float> X_d(n);
	thrust::host_vector<float> Y_d(n);
	thrust::device_vector<float> Z(n);
	
	thrust::generate(X_d.begin(), X_d.end(), [&] { return dist(rngX); });
	thrust::generate(Y_d.begin(), Y_d.end(), [&] { return dist(rngY); });

	
	thrust::device_vector<float> X = X_d;
	thrust::device_vector<float> Y = Y_d;

	if (n <= printable_n)
	{
		printf("Input data:\n");
		for (int i = 0; i < n; i++)
		printf("%f   %f\n", 1.f * X[i], 1.f * Y[i]);
		printf("\n");
	}


	float a = 2.5f;
	thrust::transform(X.begin(), X.end(), Y.begin(), Z.begin(), saxpy(a));

	if (n <= printable_n)
	{
		printf("Output data:\n");
		for (int i = 0; i < n; i++)
			printf("%f * %f + %f = %f\n", a, 1.f * X[i], 1.f * Y[i], 1.f * Z[i]);
		printf("\n");
	}

	return 0;
}
