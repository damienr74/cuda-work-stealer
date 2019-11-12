#include "gtest/gtest.h"

#include <numeric>

#include "lib/util/soa_allocator.cu.hh"

TEST(soa_allocator, representation) {
	using T0 = long double;
	using T1 = double;
	using T2 = float;
	using T3 = uint16_t;
	using T4 = uint8_t;
	constexpr uint64_t S[] = {
		sizeof(T0), sizeof(T1), sizeof(T2), sizeof(T3), sizeof(T4),
	};

	using SoA1 = StructureOfArrays<T0, T1, T2, T3>;
	constexpr auto sizes1 = SoA1::TypeSizes;
	EXPECT_EQ(S[0] + S[1] + S[2] + S[3], std::accumulate(sizes1, sizes1 + SoA1::N, 0u));
	uint64_t sorted1[] = {0, 1, 2, 3};
	for (unsigned i = 0; i < SoA1::N; i++) {
		EXPECT_EQ(sorted1[i], SoA1::Mapping[i]);
	}
	EXPECT_EQ(5*sizeof(void*), sizeof(SoA1));
	auto soa1 = SoA1(nullptr, 10);
	EXPECT_EQ(0u, (uint64_t)soa1.get<0>());
	EXPECT_EQ(10u*S[0], (uint64_t)soa1.get<1>());
	EXPECT_EQ(10u*(S[0] + S[1]), (uint64_t)soa1.get<2>());
	EXPECT_EQ(10u*(S[0] + S[1] + S[2]), (uint64_t)soa1.get<3>());

	using SoA2 = StructureOfArrays<T4, T3, T2, T1, T0>;
	constexpr auto sizes2 = SoA2::TypeSizes;
	EXPECT_EQ(S[0] + S[1] + S[2] + S[3] + S[4], std::accumulate(sizes2, sizes2 + SoA2::N, 0u));
	uint64_t sorted2[] = {4, 3, 2, 1, 0};
	for (unsigned i = 0; i < SoA2::N; i++) {
		EXPECT_EQ(sorted2[i], SoA2::Mapping[i]);
	}
	EXPECT_EQ(6*sizeof(void*), sizeof(SoA2));
	auto soa2 = SoA2(nullptr, 10);
	EXPECT_EQ(0u, (uint64_t)soa2.get<4>());
	EXPECT_EQ(10u*(S[0]), (uint64_t)soa2.get<3>());
	EXPECT_EQ(10u*(S[0] + S[1]), (uint64_t)soa2.get<2>());
	EXPECT_EQ(10u*(S[0] + S[1] + S[2]), (uint64_t)soa2.get<1>());
	EXPECT_EQ(10u*(S[0] + S[1] + S[2] + S[3]), (uint64_t)soa2.get<0>());
}
