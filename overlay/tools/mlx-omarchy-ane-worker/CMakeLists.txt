# The bounded ANE worker CLI. Compiled only for explicitly configured
# ANE device builds (MLX_OMARCHY_ANE_DEVICE): host builds carry no
# libane surface at all.
if(MLX_OMARCHY_ANE_DEVICE)
  add_executable(
    mlx-omarchy-ane-worker
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../mlx/backend/omarchy/ane/worker_libane.cpp)
  target_link_libraries(mlx-omarchy-ane-worker PRIVATE mlx ${CMAKE_DL_LIBS})
  target_include_directories(
    mlx-omarchy-ane-worker PRIVATE ${OMARCHY_ANE_INCLUDE_DIR})
  target_compile_definitions(mlx-omarchy-ane-worker
                             PRIVATE MLX_OMARCHY_ANE_DEVICE=1)
  set_target_properties(
    mlx-omarchy-ane-worker PROPERTIES OUTPUT_NAME mlx-omarchy-ane-worker
                                      INSTALL_RPATH "$ORIGIN/../lib")
  install(TARGETS mlx-omarchy-ane-worker
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
