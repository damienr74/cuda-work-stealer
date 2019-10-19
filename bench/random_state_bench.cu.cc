#include "benchmark/benchmark.h"

#include "lib/work_stealing_api.cu.hh"

static void BM_random_init(benchmark::State& state) {
	for (auto _ : state) {
		RandomState(56);
	}
}
BENCHMARK(BM_random_init);

__global__ void k_generate_random(DeviceState rs, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= rs.size) return;

	__shared__ auto rng = rs.states[tid];
	for (int i = 0; i < n; i++) {
		rng.get_uniform();
	}
	rs.states[tid] = rng;
}

static void BM_random_gen(benchmark::State& state) {
	const auto wsn = 56;
	auto rs = RandomState(wsn);
	int n = state.range();
	for (auto _ : state) {
		k_generate_random<<<wsn, 1>>>(rs.device(), n);
		CUDA_CHECK(cudaDeviceSynchronize())
			<< "error generating random data\n";
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_random_gen)->Range(1, 1<<15)->Complexity();

BENCHMARK_MAIN();
