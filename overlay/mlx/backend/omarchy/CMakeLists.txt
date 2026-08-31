target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/encoder.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/stream.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/vulkan.cpp)

# The backend dlopens libvulkan at runtime; only headers are needed at compile
# time.
if(TARGET Vulkan::Headers)
  target_link_libraries(mlx PUBLIC Vulkan::Headers)
else()
  target_include_directories(mlx PUBLIC ${Vulkan_INCLUDE_DIRS})
endif()
