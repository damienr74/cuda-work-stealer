package(default_visibility = ['//visibility:public'])

load(":cc_toolchain_config.bzl", "cc_toolchain_config")
cc_toolchain_config(name = "cuda_clang_toolchain_config")

cc_toolchain_suite(
	name="cuda_clang",
	toolchains = {
		"x86_64-ptx-61": ":cuda_clang_toolchain",
	},
)

filegroup(name = "all")

cc_toolchain(
	name = "cuda_clang_toolchain",
	toolchain_identifier = "cuda_clang_toolchain",
	toolchain_config = ":cuda_clang_toolchain_config",
	all_files = ":all",
	compiler_files = ":all",
	dwp_files = ":empty",
	linker_files = ":all",
	objcopy_files = ":empty",
	strip_files = ":empty",
	supports_param_files = 0,
)
