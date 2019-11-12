#include <functional>

#include "benchmark/benchmark.h"

#include "lib/work_stealing_api.cu.hh"


constexpr int64_t sms = 56;
using u64 = unsigned long long;
__device__ u64 collisions[sms];

static void BM_random_init(benchmark::State& state) {
	[[maybe_unused]] const auto data = RandomState(sms);
	for (auto _ : state) {
		[[maybe_unused]] const auto data = RandomState(sms);
	}
}
BENCHMARK(BM_random_init);

__global__ void k_gen_random(RandomHandles rs, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= rs.size) return;

	__shared__ auto rng = rs.handles[tid];
	for (int i = 0; i < n; i++) {
		rng.get_uniform();
	}
	// XXX: Omit writeback to have the same state for each kernel call.
	// rs.handles[tid] = rng;
}
static void BM_generate_only(benchmark::State& state) {
	auto rs = RandomState(sms);
	int n = state.range();
	for (auto _ : state) {
		k_gen_random<<<sms, 1>>>(rs.device(), n);
		CUDA_CHECK(cudaDeviceSynchronize())
			<< "error in k_gen_random\n";
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_generate_only)->Range(1, 1<<15)->Complexity();

__global__ void k_atomicAdd_no_collision(RandomHandles rs, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= rs.size) return;

	__shared__ auto rng = rs.handles[tid];
	for (int i = 0; i < n; i++) {
		atomicAdd(&collisions[tid], rng.get_uniform() % sms);
	}
	// XXX: Omit writeback to have the same state for each kernel call.
	// rs.handles[tid] = rng;
}
static void BM_atomicAdd_no_collision(benchmark::State& state) {
	auto rs = RandomState(sms);
	int n = state.range();
	for (auto _ : state) {
		k_atomicAdd_no_collision<<<sms, 1>>>(rs.device(), n);
		CUDA_CHECK(cudaDeviceSynchronize())
			<< "error in k_atomicAdd_no_collision\n";
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_atomicAdd_no_collision)->Range(1, 1<<15)->Complexity();

__global__ void k_atomicAdd_collision(RandomHandles rs, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ auto rng = rs.handles[tid];
	for (int i = 0; i < n; i++) {
		atomicAdd(&collisions[rng.get_uniform() % sms], u64(i));
	}

	// XXX: Omit writeback to have the same state for each kernel call.
	//rs.handles[tid] = rng;
}
static void BM_rng_quality(benchmark::State& state) {
	auto rs = RandomState(sms);

	int n = state.range();
	for (auto _ : state) {
		k_atomicAdd_collision<<<sms, 1>>>(rs.device(), n);
		CUDA_CHECK(cudaDeviceSynchronize())
			<< "error in k_atomicAdd_collision\n";
	}

	state.SetComplexityN(n);
}
BENCHMARK(BM_rng_quality)->Range(1, 1<<15)->Complexity();

BENCHMARK_MAIN();
