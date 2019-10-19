cc_test(
	name = "random_state",
	srcs = ["random_state_bench.cu.cc"],
	deps = [
		"//lib:work_stealer",
		"@benchmark//:benchmark",
		"@benchmark//:benchmark_main",
	],
	size = "large",
)
