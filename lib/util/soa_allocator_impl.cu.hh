#ifndef LIB_UTIL_SOA_ALLOCATOR_IMPL_CU_HH
#define LIB_UTIL_SOA_ALLOCATOR_IMPL_CU_HH

#include <numeric>
#include <cinttypes>
#include <utility>

namespace impl {

template <uint64_t N>
struct Array {
	constexpr Array() :data{} {}
	constexpr Array(uint64_t o[N]) :data{} {
		for (unsigned i = 0; i < N; i++) {
			data[i] = o[i];
		}
	}
	constexpr Array(Array<N>& o) :Array(o.data) {}
	uint64_t data[N];

	constexpr uint64_t operator[](uint64_t i) const { return data[i]; }
	constexpr uint64_t *begin() { return data; }
	constexpr const uint64_t * cbegin() const { return data; }
	constexpr uint64_t *end() { return data + N; }
	constexpr const uint64_t * cend() const { return data + N; }
};

template <uint64_t I, typename T>
struct PtrImpl {
	__host__ __device__ PtrImpl(const uint64_t *sizes, 
			const uint64_t *order, uint64_t elems, char *ptr, int n) {
		for (unsigned i = 0; i < elems; i++) {
			if (order[i] == I) break;
			ptr += n*sizes[order[i]];
		}
		val = (T*)ptr;
	}

	T *val;
	__host__ __device__ T *get() { return val; }
};

template <size_t I, typename X, typename ...Xs>
struct GetType { using type = typename GetType<I-1, Xs...>::type; };

template<typename X, typename ...Xs>
struct GetType<0, X, Xs...> { using type = X; };

template <typename Is, typename ...Ts>
struct TuplePtrImpl;

template <size_t ...Is, typename ...Ts> 
struct TuplePtrImpl<std::index_sequence<Is...>, Ts...> : public PtrImpl<Is, Ts>... {
	__host__ __device__ TuplePtrImpl(void *alloc, uint64_t n) 
		: PtrImpl<Is, Ts>(TypeSizes, Mapping.data, N, (char *)alloc, n)... {}

	template <uint64_t I, typename T = typename GetType<I, Ts...>::type>
	__host__ __device__ T *get() {
		static_assert(N > I, "Cannot get innexisting value.");
		return static_cast<PtrImpl<I, T>*>(this)->get();
	}

	template <uint64_t I, typename T = typename GetType<I, Ts...>::type>
	__host__ __device__ const T *get() const {
		return static_cast<PtrImpl<I, T>*>(this)->get();
	}

	template <uint64_t I, typename T = typename GetType<I, Ts...>::type>
	__host__ __device__ T &get(uint64_t i) { return get<I>()[i]; }


	template <uint64_t I, typename T = typename GetType<I, Ts...>::type>
	__host__ __device__ const T &get(uint64_t i) const { return get<I>()[i]; }

	constexpr static uint64_t N = sizeof...(Ts);
	constexpr static uint64_t TypeSizes[N] = {sizeof(Ts)...,};
	constexpr static uint64_t TypeAligns[N] = {alignof(Ts)...,};

	constexpr static auto sorted_idx() {
		Array<N> mapping{};
		auto &m = mapping.data;
		for (unsigned i = 0; i < N; i++) {
			m[i] = i;
		}

		for (unsigned i = 1; i < N; i++) {
			int j = i-1;
			const auto x = TypeAligns[m[i]];

			while (j >= 0 && x > TypeAligns[m[j]]) {
				m[j+1] = m[j];
				j--;
			}
			m[j+1] = i;
		}

		return mapping;
	}
	constexpr static Array<N> Mapping = sorted_idx();

	constexpr static uint64_t size_for(uint64_t n) {
		return n * std::accumulate(TypeSizes, TypeSizes + N, 0);
	}
};

}; // namespace impl

#endif // LIB_UTIL_SOA_ALLOCATOR_IMPL_CU_HH
