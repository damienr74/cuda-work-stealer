#ifndef LIB_INIT_WS_STATE_CU_HH
#define LIB_INIT_WS_STATE_CU_HH

#include <algorithm>
#include <curand_kernel.h>
#include <random>

#include "lib/util/cuda_check.hh"


struct RandomHandle {
	curandState_t state;
	uint64_t seed;

	__device__ void init(int64_t tid) {
		curand_init(seed, 1, tid, &state);
	}
	__device__ uint32_t get_uniform() { return curand(&state); }
};

struct RandomHandles {
	const int64_t size;
	RandomHandle * __restrict__ const handles;
};


class RandomState {
public:
	__host__ RandomState(int64_t n) : n_{n}, randstates_{nullptr} {
		CUDA_CHECK(cudaMalloc(&randstates_, n * sizeof *randstates_))
			<< "could not allocate random state data\n";

		std::vector<uint64_t> seeds(2*n);
		std::iota(seeds.begin(), seeds.begin()+n, 1);
		std::seed_seq(seeds.cbegin(), seeds.cbegin() + n)
			.generate(seeds.begin() + n, seeds.end());

		CUDA_CHECK(cudaMalloc(&randstates_, n * sizeof *randstates_))
			<< "could not allocate random state data\n";

		uint64_t *device_seeds = nullptr;
		CUDA_CHECK(cudaMalloc(&device_seeds, n * sizeof *randstates_))
			<< "could not allocate random state data\n";
		CUDA_CHECK(cudaMemcpy(device_seeds, seeds.data()+n, 
			n * sizeof *device_seeds, cudaMemcpyHostToDevice))
			<< "error copying state to device\n";

		__global__ void k_random_state_seed(
			RandomHandle *states,
			uint64_t *seeds,
			int64_t n);

		k_random_state_seed<<<n_, 1>>>(randstates_, device_seeds, n);
		CUDA_CHECK(cudaDeviceSynchronize())
			<< "error seeding state\n";
		CUDA_CHECK(cudaFree(device_seeds))
			<< "error freeing intermediate seed buffer\n";
	}

	// Can be called multiple times.
	void reinit() {
		__global__ void k_random_state_reinit(
			RandomHandle *states,
			int64_t n);

		k_random_state_reinit<<<n_, 1>>>(randstates_, n_);
		CUDA_CHECK(cudaDeviceSynchronize())
			<< "error calling reinit\n";
	}

	__host__ RandomHandles device() const {
		return {n_, randstates_};
	}

	__host__ __device__ int64_t size() const { return n_; }

	~RandomState() {
		CUDA_CHECK(cudaFree(randstates_))
			<< "could not free random state data\n";
	}
private:
	int64_t n_;
	RandomHandle *randstates_;
};

#endif // LIB_INIT_WS_STATE_CU_HH
