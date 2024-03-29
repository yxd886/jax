load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)


# https://github.com/bazelbuild/bazel-skylib/releases
http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
    ],
)

# To update TensorFlow to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
#http_archive(
#    name = "org_tensorflow",
#    sha256 = "c868259c92a669743f77c6cdc450a1b92bf01ac9f814825264e6adebd0acdd6e",
#    strip_prefix = "tensorflow-4ace926c663be8ecd13505cd4316b7b3380008af",
#    urls = [
#        "https://github.com/tensorflow/tensorflow/archive/4ace926c663be8ecd13505cd4316b7b3380008af.tar.gz",
#    ],
#)

# For development, one can use a local TF repository instead.
local_repository(
    name = "org_tensorflow",
    path = "../tensorflow",
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace", "tf_bind")

tf_workspace(
    path_prefix = "",
    tf_repo_name = "org_tensorflow",
)

tf_bind()

# Required for TensorFlow dependency on @com_github_grpc_grpc

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)

apple_rules_dependencies()

load(
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)

apple_support_dependencies()

load("@upb//bazel:repository_defs.bzl", "bazel_version_repository")

bazel_version_repository(name = "bazel_version")



new_local_repository(
name = "python_linux",
path = "/home/net/.conda/envs/jax",
build_file_content = """
cc_library(
name = "python37-lib",
srcs = ["lib/libpython3.7m.so"],
hdrs = glob(["include/python3.7m/*.h"]),
includes = ["include/python3.7m"],
visibility = ["//visibility:public"],
)
"""
)


new_local_repository(
name = "openmpi",
path = "/usr/local/openmpi",
build_file_content = """
cc_library(
name = "openmpi-lib",
srcs = ["lib/libmpi.so","lib/libmpi_mpifh.so"],
hdrs = glob(["include/*.h"]),
includes = ["include"],
visibility = ["//visibility:public"],
)
"""
)

new_local_repository(
name = "opencurl",
path = "/usr/",
build_file_content = """
cc_library(
name = "curl-lib",
srcs = ["lib/x86_64-linux-gnu/libcurl.so"],
hdrs = glob(["include/x86_64-linux-gnu/curl/*.h"]),
includes = ["include/x86_64-linux-gnu/curl"],
visibility = ["//visibility:public"],
)
"""
)