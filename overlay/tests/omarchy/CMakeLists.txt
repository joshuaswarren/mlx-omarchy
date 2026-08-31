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
