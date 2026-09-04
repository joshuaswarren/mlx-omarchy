add_executable(omarchy_runtime_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_runtime.cpp)
target_link_libraries(omarchy_runtime_tests PRIVATE mlx doctest)
target_compile_options(omarchy_runtime_tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_runtime_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_runtime_tests)
add_test(NAME omarchy_runtime_tests COMMAND omarchy_runtime_tests)

add_executable(omarchy_copy_offset_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_copy_offsets.cpp)
target_link_libraries(omarchy_copy_offset_tests PRIVATE mlx doctest)
target_compile_options(omarchy_copy_offset_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_copy_offset_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_copy_offset_tests)
add_test(NAME omarchy_copy_offset_tests COMMAND omarchy_copy_offset_tests)

add_executable(omarchy_primitive_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_primitives.cpp)
target_link_libraries(omarchy_primitive_tests PRIVATE mlx doctest)
target_compile_options(omarchy_primitive_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_primitive_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_primitive_tests)
add_test(NAME omarchy_primitive_tests COMMAND omarchy_primitive_tests)

add_executable(omarchy_ane_bundle_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/ane/test_bundle.cpp)
target_link_libraries(omarchy_ane_bundle_tests PRIVATE mlx doctest)
target_compile_options(omarchy_ane_bundle_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_ane_bundle_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_ane_bundle_tests)
add_test(NAME omarchy_ane_bundle_tests COMMAND omarchy_ane_bundle_tests)

add_executable(omarchy_kv_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_kv_ops.cpp)
target_link_libraries(omarchy_kv_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_kv_ops_tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_kv_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_kv_ops_tests)

add_test(NAME omarchy_kv_ops_tests COMMAND omarchy_kv_ops_tests)

add_executable(omarchy_error_contract_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_error_contract.cpp)
target_link_libraries(omarchy_error_contract_tests PRIVATE mlx doctest)
target_compile_options(omarchy_error_contract_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_error_contract_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_error_contract_tests)
add_test(NAME omarchy_error_contract_tests
         COMMAND omarchy_error_contract_tests)

add_executable(omarchy_shape_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_shape_ops.cpp)
target_link_libraries(omarchy_shape_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_shape_ops_tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_shape_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_shape_ops_tests)
add_test(NAME omarchy_shape_ops_tests COMMAND omarchy_shape_ops_tests)

add_executable(omarchy_reduce_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce_ops.cpp)
target_link_libraries(omarchy_reduce_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_reduce_ops_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_reduce_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_reduce_ops_tests)
add_test(NAME omarchy_reduce_ops_tests COMMAND omarchy_reduce_ops_tests)

add_executable(omarchy_indexing_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_indexing_ops.cpp)
target_link_libraries(omarchy_indexing_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_indexing_ops_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_indexing_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_indexing_ops_tests)
add_test(NAME omarchy_indexing_ops_tests COMMAND omarchy_indexing_ops_tests)

add_executable(omarchy_matmul_family_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_matmul_family.cpp)
target_link_libraries(omarchy_matmul_family_tests PRIVATE mlx doctest)
target_compile_options(omarchy_matmul_family_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_matmul_family_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_matmul_family_tests)
add_test(NAME omarchy_matmul_family_tests
         COMMAND omarchy_matmul_family_tests)

add_executable(omarchy_distributed_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_ops.cpp)
target_link_libraries(omarchy_distributed_tests PRIVATE mlx doctest)
target_compile_options(omarchy_distributed_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_distributed_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_distributed_tests)
add_test(NAME omarchy_distributed_tests
         COMMAND omarchy_distributed_tests)

add_executable(omarchy_compiled_tape_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_compiled_tape.cpp)
target_link_libraries(omarchy_compiled_tape_tests PRIVATE mlx doctest)
target_compile_options(omarchy_compiled_tape_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_compiled_tape_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_compiled_tape_tests)
add_test(NAME omarchy_compiled_tape_tests
         COMMAND omarchy_compiled_tape_tests)

add_executable(omarchy_fft_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_fft_ops.cpp)
target_link_libraries(omarchy_fft_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_fft_ops_tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_fft_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_fft_ops_tests)
add_test(NAME omarchy_fft_ops_tests COMMAND omarchy_fft_ops_tests)

add_executable(omarchy_fast_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_fast_ops.cpp)
target_link_libraries(omarchy_fast_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_fast_ops_tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_fast_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_fast_ops_tests)
add_test(NAME omarchy_fast_ops_tests COMMAND omarchy_fast_ops_tests)

add_executable(omarchy_linalg_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_linalg_ops.cpp)
target_link_libraries(omarchy_linalg_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_linalg_ops_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_linalg_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_linalg_ops_tests)
add_test(NAME omarchy_linalg_ops_tests COMMAND omarchy_linalg_ops_tests)
add_executable(omarchy_eig_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_eig_ops.cpp)
target_link_libraries(omarchy_eig_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_eig_ops_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_eig_ops_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_eig_ops_tests)
add_test(NAME omarchy_eig_ops_tests COMMAND omarchy_eig_ops_tests)


add_executable(omarchy_take_fill_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_take_negative_fill.cpp)
target_link_libraries(omarchy_take_fill_tests PRIVATE mlx doctest)
target_compile_options(omarchy_take_fill_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_take_fill_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_take_fill_tests)
add_test(NAME omarchy_take_fill_tests COMMAND omarchy_take_fill_tests)
# test_eq_math and test_sdpa_norm_regression are still owned by their
# respective authors. omarchy_topk_ops_tests is UNREGISTERED (2026-09-02,
# IndexingTopkRepair): its single case only exercised WideRowTopK's
# reverted wide-row kernel, and the kernel's shader source was lost in
# the stash sweep, so the suite had nothing real to test. The wide-row
# capability gap is pinned by the refusal test in test_indexing_ops.cpp
# ("argpartition wide rows keep the named refusal..."), which flips to a
# value test when a correct selection kernel lands. Placeholder stubs
# that assert nothing are forbidden: unregister the target instead.
# See hub IRC Main 2026-09-02 for the no-placeholder rule.

add_executable(omarchy_conv_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_conv_general.cpp)
target_link_libraries(omarchy_conv_tests PRIVATE mlx doctest)
target_compile_options(omarchy_conv_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_conv_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_conv_tests)
add_test(NAME omarchy_conv_tests COMMAND omarchy_conv_tests)

add_executable(omarchy_complex_ops_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_complex_ops.cpp)
target_link_libraries(omarchy_complex_ops_tests PRIVATE mlx doctest)
target_compile_options(omarchy_complex_ops_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_complex_ops_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_complex_ops_tests)
add_test(NAME omarchy_complex_ops_tests
         COMMAND omarchy_complex_ops_tests)

add_executable(omarchy_select_layout_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_select_ops.cpp)
target_link_libraries(omarchy_select_layout_tests PRIVATE mlx doctest)
target_compile_options(omarchy_select_layout_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_select_layout_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_select_layout_tests)
add_test(NAME omarchy_select_layout_tests
         COMMAND omarchy_select_layout_tests)

add_executable(omarchy_fft_general_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_fft_general.cpp)
target_link_libraries(omarchy_fft_general_tests PRIVATE mlx doctest)
target_compile_options(omarchy_fft_general_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_fft_general_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_fft_general_tests)
add_test(NAME omarchy_fft_general_tests COMMAND omarchy_fft_general_tests)

add_executable(omarchy_fast_regression_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_sdpa_norm_regression.cpp)
target_link_libraries(omarchy_fast_regression_tests PRIVATE mlx doctest)
target_compile_options(omarchy_fast_regression_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_fast_regression_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_fast_regression_tests)
add_test(NAME omarchy_fast_regression_tests COMMAND omarchy_fast_regression_tests)

# ScatterDeterminism: float scatter reductions (hardware fp32 atomicAdd
# for Sum, CAS fp32 multiply for Prod, fp32 scratch for half types),
# bool Scatter (packed-byte deterministic RMW), and non-dense layout
# materialization for indices and updates.
add_executable(omarchy_scatter_determinism_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_scatter_determinism.cpp)
target_link_libraries(omarchy_scatter_determinism_tests PRIVATE mlx doctest)
target_compile_options(omarchy_scatter_determinism_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_scatter_determinism_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_scatter_determinism_tests)
add_test(NAME omarchy_scatter_determinism_tests
         COMMAND omarchy_scatter_determinism_tests)

# FixEqualNanAndMath: W2 equal_nan flag plumbing on Equal + the bool
# comparison sextet through logical_or.comp, W5 log10 exactness
# (f32-exact powers of ten), and W8 sin/cos/tan on the common and
# full-magnitude paths.
add_executable(omarchy_eq_math_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_eq_math.cpp)
target_link_libraries(omarchy_eq_math_tests PRIVATE mlx doctest)
target_compile_options(omarchy_eq_math_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_eq_math_tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_eq_math_tests)
add_test(NAME omarchy_eq_math_tests COMMAND omarchy_eq_math_tests)

# FuseDecodeChains: decode dispatch counter harness (host-side trace
# counters over a faithful Qwen2.5-0.5B decode step) and fused-chain
# compiled-versus-eager equivalence tests.
add_executable(omarchy_decode_dispatch_count
               ${CMAKE_CURRENT_SOURCE_DIR}/decode_dispatch_count.cpp)
target_link_libraries(omarchy_decode_dispatch_count PRIVATE mlx doctest)
target_compile_options(omarchy_decode_dispatch_count
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_decode_dispatch_count
                    PRIVATE ${SANITIZER_LINK_FLAGS})
add_test(NAME omarchy_decode_dispatch_count
         COMMAND omarchy_decode_dispatch_count)

# FuseDecodeChains: fused elementwise-chain compiled-versus-eager
# equivalence tests plus dispatch-collapse and bf16-refusal guards.
add_executable(omarchy_fused_chain_tests
               ${CMAKE_CURRENT_SOURCE_DIR}/test_fused_chain.cpp)
target_link_libraries(omarchy_fused_chain_tests PRIVATE mlx doctest)
target_compile_options(omarchy_fused_chain_tests
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_fused_chain_tests
                    PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(omarchy_fused_chain_tests)
add_test(NAME omarchy_fused_chain_tests COMMAND omarchy_fused_chain_tests)

# TwoRankDistributed: the five distributed primitives value-proven on a
# real two-rank ring group over loopback. The doctest binary is ONE rank's
# process and fails its own group-size guards when run alone, so the
# registered test is the launcher script, which runs ranks 0 and 1 against
# a localhost hostfile and requires both green. No
# doctest_discover_tests here: discovery would run the binary without
# MLX_RANK/MLX_HOSTFILE and fail by design.
add_executable(omarchy_two_rank_harness
               ${CMAKE_CURRENT_SOURCE_DIR}/distributed/test_two_rank.cpp)
target_link_libraries(omarchy_two_rank_harness PRIVATE mlx doctest)
target_compile_options(omarchy_two_rank_harness
                       PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(omarchy_two_rank_harness PRIVATE ${SANITIZER_LINK_FLAGS})

add_test(NAME omarchy_two_rank_harness
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/distributed/run-two-rank.sh)
