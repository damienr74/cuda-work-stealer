#include "gtest/gtest.h"
#include "lib/api.hh"

TEST(work_stealing_size, compute_static) {
	struct State { uint8_t data[256]; };
	using WorkIterator = uint8_t;

	cudaDeviceProp props;
	CUDA_CHECK(cudaGetDeviceProperties(&props, 0))
		<< "could not communicate with device..\n";

	auto ws = WorkStealer<std::size_t>{OverloadSet{
		[](GroupSize) {return 32;},
		[](StateSize) {return sizeof(State);},
		[](WorkState) {return sizeof(WorkIterator);},
		[](auto) {return -1;}
	}};
	EXPECT_GT(ws.instances(), 0);
	EXPECT_EQ(ws.group_size(), 32);
	EXPECT_LE(ws.group_hint(), 1024/32);
	EXPECT_EQ(ws.state_size(), int64_t(sizeof(State)));
	EXPECT_EQ(ws.work_state(), int64_t(sizeof(WorkIterator)));
	EXPECT_EQ(ws.valid(), true);

	ws = WorkStealer<std::size_t>{OverloadSet{
		[](GroupSize) {return 32;},
		[](StateSize) {return (1<< 15) * sizeof(State);},
		[](WorkState) {return sizeof(WorkIterator);},
		[](auto) {return -1;}
	}};
	EXPECT_GT(ws.instances(), 0);
	EXPECT_EQ(ws.group_size(), 32);
	EXPECT_EQ(ws.group_hint(), 24);
	EXPECT_EQ(ws.state_size(), int64_t((1 << 15) *sizeof(State)));
	EXPECT_EQ(ws.work_state(), int64_t(sizeof(WorkIterator)));
	EXPECT_EQ(ws.valid(), true);

	EXPECT_EQ(ws.valid(), true);

	ws = WorkStealer<std::size_t>{OverloadSet{
		[](GroupSize) {return 32;},
		[](GroupHint) {return 32;},
		[](StateSize) {return (1<< 15) * sizeof(State);},
		[](WorkState) {return sizeof(WorkIterator);},
		[](auto) {return -1;}
	}};
	EXPECT_LE(ws.group_hint(), 24);
	EXPECT_EQ(ws.valid(), true);

	ws = WorkStealer<std::size_t>{OverloadSet{
		[](auto) {return -1;}
	}};
	EXPECT_EQ(ws.valid(), false);
}
