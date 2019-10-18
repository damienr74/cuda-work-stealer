#ifndef WORK_SHARING_API_CUH
#define WORK_SHARING_API_CUH

#include <cinttypes>
#include <string_view>
#include <curand.h>

#include "lib/util/cuda_check.hh"
#include "lib/util/overload_set.hh"
#include "lib/util/fmt_quantity.hh"

// These scturtures define a dummy type used by the Config OverloadSet.
struct Instances{}; // Suggested number of thread blocks.
struct GroupSize{}; // Number of threads per logical unit of work.
struct GroupHint{}; // Hinted number of groups per SM if memory allows.
struct StateSize{}; // Size of the state used for one group.
struct WorkState{}; // Size of the state to encode work stealing.

template<typename Work>
class WorkStealer {
public:
	template<typename Config>
	WorkStealer(Config cfg)
		: props_{}
		, init_props_{(CUDA_CHECK(cudaGetDeviceProperties(&props_, 0)) <<
				"could not communicate with device..\n", 1)}
		, instances_{cfg(Instances{}) > 0 ? cfg(Instances{}) :
				(props_.maxThreadsPerMultiProcessor /
				 props_.maxThreadsPerBlock)
					*props_.multiProcessorCount}
       		, group_size_{cfg(GroupSize{})}
		, group_hint_{cfg(GroupHint{})}
		, state_size_{cfg(StateSize{})}
		, work_state_{cfg(WorkState{})}
		, valid_{fix_configuration()}
	{
		if (valid_) {
			std::cout << "Launching " << instances_
				<< " global workstealing queues with "
			       	<< group_hint_ << " groups of size "
				<< group_size_ << ". Total Memory usage: "
				<< fmt_quantity(total_mem()) << "B\n";
		}
	}

	constexpr int64_t instances() const { return instances_; }
	constexpr int64_t group_size() const { return group_size_; }
	constexpr int64_t group_hint() const { return group_hint_; }
	constexpr int64_t state_size() const { return state_size_; }
	constexpr int64_t work_state() const { return work_state_; }
	constexpr bool valid() const { return valid_; }
	constexpr int64_t total_mem() const {
		return instances_ *
			(work_state_ + group_hint_*state_size_);
	}


private:

	constexpr bool fix_configuration() {
		using namespace std::literals;
		constexpr auto invalid_helper =
			[](int64_t prop, std::string_view prop_name) -> bool{
			const bool invalid = prop < 1;
			if (invalid) {
				std::cerr << "Invalid " << prop_name << '('
					<< prop << ")\n";
			}
			return invalid;
		};
		
		const bool invalid = invalid_helper(group_size_, "group_size"sv) |
		       invalid_helper(state_size_, "state_size"sv) |
		       invalid_helper(work_state_, "work_state"sv);

		const auto invalid_memory = [&]() -> bool {
			return total_mem() > int64_t(props_.totalGlobalMem);
		};

		std::cerr << "requires more memory than is available on device.\n";
		while (invalid_memory() || !invalid) {
			if (!invalid && (group_hint_ <= 0 || invalid_memory())) {
				// globalMem >= instances(work_state + unknown*state_size)
				// globalMem/instances >= work_state + unknown*state_size
				// globalMem/instances - work_state >= unknown*state_size
				// (globalMem/instances - work_state)/state_size >= unknown
				const int64_t per_instance = props_.totalGlobalMem/instances_;
				group_hint_ = (per_instance - work_state_)/state_size_;
				group_hint_ = std::min(props_.maxThreadsPerBlock/group_size_,
						group_hint_);
			}
			if (!invalid_memory()) break;
			instances_ -= 2;
		}
		return !invalid;
	}

	cudaDeviceProp props_;
	int64_t init_props_;
	int64_t instances_;
	int64_t group_size_;
	int64_t group_hint_;
	int64_t state_size_;
	int64_t work_state_;
	bool valid_;
	curandStatus *states; // allocate # of instances.
};

#endif // WORK_SHARING_API_CUH
