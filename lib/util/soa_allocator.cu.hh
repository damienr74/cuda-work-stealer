#ifndef LIB_UTIL_SOA_ALLOCATOR_CU_HH
#define LIB_UTIL_SOA_ALLOCATOR_CU_HH

#include "lib/util/soa_allocator_impl.cu.hh"

// Use this as a tuple of pointers that gets a single pointer that allocates
// the data for a structure of arrays with n items.  The amount of memory
// can be obtained from StructureOfArrays<...>::size_for(n).
//
// Using this has some benefits such as minimizing padding (most of the time)
// and needing only one allocation per set of structure of arrays.
//
// The disadvantage is that the structure fields are not named (think tuple).
//
// An element can be obtained as follows:
// 	StructureOfArrays<bool, double> array(mem, n);
// 	array.get<0>(tid) = false;
// 	array.get<0>()[tid] = false;
// 	array.get<1>(tid) = sin(pi/2 - eps);;
template <typename ...Ts>
struct StructureOfArrays 
	: impl::TuplePtrImpl<std::make_index_sequence<sizeof...(Ts)>, Ts...> {
	
	uint64_t n;

	StructureOfArrays(void *alloc, uint64_t n)
		: impl::TuplePtrImpl<
			std::make_index_sequence<sizeof...(Ts)>
			, Ts...
		>(alloc, n)
		, n{n}
	{}
};

#endif // LIB_UTIL_SOA_ALLOCATOR_CU_HH
