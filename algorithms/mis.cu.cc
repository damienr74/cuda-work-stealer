// Implementation of 1.2905^n algorithm for Maximum Independent Set
//
//  mis(G)
//    if there exists a vertex with degree 0
//      return 1 + mis(G\v)
//    if there exists a vertex with degree 1
//      return 1 + mis(G\N[v])
//    if maximum degree > 2 then
//      return max(1 + mis(G\N[v]), mis(G\v))
//
//    union of disjoint cycles (doesn't matter)

#include "lib/util/soa_allocator.cu.hh"

constexpr auto block(uint64_t idx) -> uint64_t {
	return (idx + 63) / 64;
};

#pragma message("review mem_impl for shared vs registers vs global.")
template <uint64_t Ts>
struct MaximumIndependentSetState {
	uint64_t * __restrict__ graph;
	uint8_t * __restrict__ trace; 
	uint8_t * __restrict__ trace_ops;
	uint8_t trace_length;
	uint8_t trace_ops_length;
	uint8_t n;

	__device__ void delete_vertex(uint64_t tid, uint8_t vertex) {
		// mem_impl
		trace_ops[trace_ops_length++] = 1;
		remove(tid, vertex);
	}

	__device__ void delete_neighbourhood(uint64_t tid, uint8_t vertex) {
		const uint64_t blocks = block(n);
		const uint64_t t = tid % Ts;
		// mem_impl
		trace_ops_length++;
		for (uint64_t i = 0; i < blocks; i++) {
			uint64_t vs = graph[blocks * block(vertex) + i];
			if (t == 0) trace_ops[trace_ops_length] += __popc(vs);
			while (vs) {
				const uint64_t bit = vs ^ (vs & (vs - 1));
				const uint64_t next = __ffs(bit) - 1 + i * 64;
				remove(tid, next);
				vs &= vs - 1;
			}
		}
	}

	__device__ void restore_operation(uint64_t tid) {
		// mem_impl
		const auto op_cnt = trace_ops[trace_ops_length--];
		for (uint64_t i = 0; i < op_cnt; i++) {
			insert(tid, trace[trace_length]);
		}
	}

	template <typename Op>
	__device__ void update_vertex(uint64_t tid, uint8_t vertex, Op && op) {
		const uint64_t t = tid % Ts;
		const uint64_t blocks = block(n);

		for (int i = 0; i < blocks; i++) {
			uint64_t g = graph[blocks * block(vertex) + i];
			for (uint64_t j = t; j < n; j += Ts) {
				uint64_t mask = g & (1ull << t);
				if (mask) {
					op(graph[blocks * block(j) + i], 1ull << t);
				}
			}
		}
	}

	__device__ void remove(uint64_t tid, uint64_t vertex) {
		constexpr auto remove = [](uint64_t &v, uint64_t update) {
			v &= ~update;
		};

		// mem_impl
		trace[++trace_length] = vertex;
		update_vertex(tid, vertex, remove);
	}

	__device__ void insert(uint64_t tid, uint8_t vertex) {
		constexpr auto insert = [](uint64_t &v, uint64_t update) {
			v |= update;
		};

		update_vertex(tid, vertex, insert);
	}
};

int main() {}
