cc_library(
	name = "work_stealer",
	deps = [
		"//lib/util",
		"//lib:curand_adapter",
	],
	hdrs = ["work_stealing_api.cu.hh"],
	visibility = ["//visibility:public"],
)

cc_library(
	name = "curand_adapter",
	deps = ["//lib/util"],
	hdrs = ["init_ws_state.cu.hh"],
	srcs = ["init_ws_state.cu.cc"],
	visibility = ["//lib:__pkg__"],
)
