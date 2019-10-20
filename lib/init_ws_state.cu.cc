#include "lib/init_ws_state.cu.hh"

__global__ void k_random_state_seed(State *states, uint64_t *seeds, int64_t n) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= n) return;

	auto state = states[tid];
	state.seed = seeds[tid];
	state.init(tid);
	states[tid] = state;
}

__global__ void k_random_state_reinit(State *states, int64_t n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= n) return;

	auto state = states[tid];
	state.init(tid);
	states[tid] = state;
}
