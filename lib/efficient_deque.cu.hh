#ifndef EFFICIENT_DEQUE_CU_HH
#define EFFICIENT_DEQUE_CU_HH

#include "lib/util/cuda_check.hh"

template <class T>
class Optional {
	T t;
	bool valid = false;

public:
	Optional(T &t): t{t}, valid{true} {}
	Optional(): t{}, valid{false} {}

	bool operator()() const { return valid; }
	T *operator->() const { return &t; }
};

template <class T>
inline __device__ T atomic_relaxed(T *addr) {
	const volatile T * vaddr = addr;
	const T v = *vaddr;
	return v;
}

template <class T>
inline __device__ T atomic_aquire(T *addr) {
	const volatile T * vaddr = addr;
	const T v = *vaddr;
	__threadfence();
	return v;
}

template <class T>
inline __device__ T atomic_sc(T *addr) {
	const volatile T *vaddr = addr;
	__threadfence();
	const T v = *vaddr;
	__threadfence();
	return v;
}

template <class T>
inline __device__ void atomic_store_relaxed(T *addr, T value) {
	volatile T *vaddr = addr;
	__threadfence();
	*vaddr = value;
}

template <class T>
inline __device__ void atomic_store_sc(T *addr, T value) {
	__threadfence();
	volatile T *vaddr = addr;
	__threadfence();
	*vaddr = value;
	__threadfence();
}

template <class Work>
class StealingDeque {
	Work * __restrict__ ptr;
	int64_t bottom;
	int64_t top;
	int64_t cap;
public:
	__device__ void push(Work n) {
		auto b = atomic_relaxed(&bottom);
		auto t = atomic_aquire(&top);
		auto a = atomic_relaxed(&ptr);
		a[b & cap] = n;
		atomic_store_relaxed(bottom, bottom + 1);
	}


	__device__ Optional<Work> pop() {
		auto b = atomic_relaxed(&bottom);
		auto t = atomic_relaxed(&top);

		if (b - t <= 0) {
			return {};
		}

		b = b - 1;
		atomic_store_relaxed(&bottom, b);
		__threadfence();
		t = atomic_relaxed(&top);

		auto size = b - t;
		if (size < 0) {
			atomic_store_relaxed(&bottom, b + 1);
			return {};
		}

		auto a = atomic_relaxed(ptr);
		auto data = a[b & cap];

		if (size != 0) {
			return data;
		}

		if (atomicCAS(&t, t, t + 1) == t) {
			atomic_store_relaxed(&t, t + 1);
			return data;
		} 

		atomic_store_relaxed(&t, t + 1);
		return {};
	}

	__device__ Optional<Work> steal() {
		auto t = atomic_aquire(&top);
		__threadfence();
		auto b = atomic_aquire(&bottom);

		auto size = b - t;
		if (size <= 0) {
			return {};
		}

		auto a = atomic_aquire(&ptr);
		auto data = a[t & cap];

		if (atomicCAS(&top, t, t + 1) == t) {
			return data;
		}
		return {};
	}
};

template <class Work>
struct StealingDeques {
	StealingDeque<Work> * deques;
	int64_t size;
};

template <class Work>
class WorkStealingState {
	StealingDeques<Work> state;
	Work *work_buf;
public:
	WorkStealingState<Work>(int64_t n, int64_t cap) 
		: state{nullptr, n}
		, work_buf{nullptr} {

		CUDA_CHECK(cudaMalloc(&state.deques, n * sizeof *state.deques))
				<< "allocating work queues\n";

		CUDA_CHECK(cudaMalloc(&work_buf, (cap + n) * sizeof *work_buf))
				<< "allocating work bufffer\n";
	}

	StealingDeques<Work> device() {
		return state;
	}

	~WorkStealingState<Work>() {
		CUDA_CHECK(cudaFree(state.deques)) << "freeing work states\n";
		CUDA_CHECK(cudaFree(work_buf)) << "freeing work states\n";
	}
};

#endif // EFFICIENT_DEQUE_CU_HH
