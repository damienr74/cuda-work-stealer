#include "lib/work_stealing_api.cu.hh"

__device__ bool complete;
__device__ volatile uint64_t yield = 0;

__device__ void pseudo_yeild() {
	yield += yield;
}

template <class Work>
__global__ void work_stealing_scheduler(StealingDeques<Work> deques, RandomHandles rands) {
	complete = false;
	const uint64_t sid = blockIdx.x;
	const uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	RandomHandle handle = rands.handles[sid];
	StealingDeque<Work> &deque = deques.deques[sid];

	using Continuation = decltype(Work{}.execute());

	Optional<Work> work{};
	if (tid == 0) {
		// Assign work to root.
	}


	while (!complete) {
		if (work) {
			Continuation cont = work.execute();
			switch (cont.num) {
			case 0:
				work = deque.pop();
				break;
			case 1:
				work = cont.c1();
				break;
			case 2:
				deque.push(cont.c1());
				work = cont.c2();
				break;
			}
		} else {
			pseudo_yeild();
			auto vid = handle.get_uniform() % deques.size;
			auto &victim = deques.deques[vid];
			work = victim.steal();
		}
	}
}
