cc_library(
	name = "work_stealer",
	deps = [
		"//lib/util",
		"//lib:curand_adapter",
		"//lib:efficient_deque",
	],
	hdrs = ["work_stealing_api.cu.hh"],
	srcs = ["work_stealing_impl.cu.cc"],
	visibility = ["//visibility:public"],
)

cc_library(
	name = "efficient_deque",
	hdrs = ["efficient_deque.cu.hh"],
	visibility = ["//visibility:public"],
)

cc_library(
	name = "curand_adapter",
	deps = ["//lib/util"],
	hdrs = ["init_ws_state.cu.hh"],
	srcs = ["init_ws_state.cu.cc"],
	visibility = ["//lib:__pkg__"],
)
