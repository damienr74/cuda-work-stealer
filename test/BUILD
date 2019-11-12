cc_test(
	name = "work_stealing_size",
	srcs = [
		"work_stealing_size.cc",
		"util_soa_allocator.cu.cc",
	],
	copts = ["-Iexternal/gtest/include"],
	deps = [
		"@gtest//:main",
		"//lib:work_stealer",
	],
	size = "small",
)
