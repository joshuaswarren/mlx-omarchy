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
