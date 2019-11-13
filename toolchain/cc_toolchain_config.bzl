load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
     "feature",
     "flag_group",
     "flag_set",
     "tool_path")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_skylib//lib:paths.bzl", "paths")


def _impl(ctx):
  clang_path = "/usr/local/bin/clang++"
  tool_paths = [
    tool_path(
      name = "gcc",
      path = clang_path,
    ),
    tool_path(
      name = "ld",
      path = "/usr/bin/ld",
    ),
    tool_path(
      name = "ar",
      path = "/usr/local/bin/llvm-ar",
    ),
    tool_path(
      name = "cpp",
      path = clang_path,
    ),
    tool_path(
      name = "gcov",
      path = "/bin/false",
    ),
    tool_path(
      name = "nm",
      path = "/usr/local/bin/llvm-nm",
    ),
    tool_path(
      name = "objdump",
      path = "/usr/bin/cuobjdump",
    ),
    tool_path(
      name = "strip",
      path = "/usr/local/bin/llvm-strip",
    ),
  ]


  toolchain_compile_features = feature(
    name = "toolchain_compiler_features",
    enabled = True,
    flag_sets = [
      flag_set(
        actions = [
          ACTION_NAMES.assemble,
	  ACTION_NAMES.preprocess_assemble,
	  ACTION_NAMES.linkstamp_compile,
	  ACTION_NAMES.c_compile,
	  ACTION_NAMES.cpp_compile,
	  ACTION_NAMES.cpp_header_parsing,
	  ACTION_NAMES.cpp_module_compile,
	  ACTION_NAMES.cpp_module_codegen,
	  ACTION_NAMES.lto_backend,
	  ACTION_NAMES.clif_match,
        ],
        flag_groups = [
          flag_group(flags = [
	    "-std=c++17",
	    "-xcuda",
	    "--cuda-gpu-arch=sm_70",
	    "-Wall",
	    "-Wextra",
	    "-pedantic",
	    "-fno-exceptions",
	  ]),
        ],
      ),
    ],
  )
  toolchain_link_features = feature(
    name = "toolchain_link_features",
    enabled = True,
    flag_sets = [
      flag_set(
        actions = [
	  ACTION_NAMES.lto_backend,
	  ACTION_NAMES.clif_match,
	  "c++-link-executable",
	  "c++-link-dynamic-library",
	  "c++-link-nodeps-dynamic-library",
        ],
	flag_groups = [
          flag_group(flags = [
            "-L/usr/local/cuda/lib64",
	    "-lcudart_static",
	    "-ldl",
	    "-lrt",
	    "-lpthread",
	  ]) ,
        ]
      ),
    ]
  )
  
  return cc_common.create_cc_toolchain_config_info(
    ctx = ctx,
    toolchain_identifier = "x86_64-ptx-61",
    host_system_name = "i686-unknown-linux-gnu",
    target_system_name = "cuda-unknown-ptx",
    target_cpu = "x86_64-ptx-61",
    target_libc = "unknown",
    compiler = "clang++-10",
    abi_version = "unknown",
    abi_libc_version = "unknown",
    tool_paths = tool_paths,
    features = [
      toolchain_link_features,
      toolchain_compile_features,
    ],
    cxx_builtin_include_directories = [
      paths.normalize("/usr/local/lib/clang/10.0.0/include/cuda_wrappers"),
      paths.normalize("/usr/local/cuda-10.1/include"),
      paths.normalize("/usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../../../include/c++/7.4.0"),
      paths.normalize("/usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../../../include/x86_64-linux-gnu/c++/7.4.0"),
      paths.normalize("/usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../../../include/c++/7.4.0/backward"),
      paths.normalize("/usr/local/include"),
      paths.normalize("/usr/local/lib/clang/10.0.0/include"),
      paths.normalize("/usr/include/x86_64-linux-gnu"),
      paths.normalize("/usr/include"),
    ]
  )

cc_toolchain_config = rule(
	implementation = _impl,
	attrs = {},
	provides = [CcToolchainConfigInfo],
)
