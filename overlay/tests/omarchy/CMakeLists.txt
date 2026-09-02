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
