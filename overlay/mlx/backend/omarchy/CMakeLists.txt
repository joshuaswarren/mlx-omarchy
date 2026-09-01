find_program(OMARCHY_GLSLC glslc)
find_program(OMARCHY_GLSLANG glslangValidator)
if(NOT OMARCHY_GLSLC AND NOT OMARCHY_GLSLANG)
  message(
    FATAL_ERROR
      "Neither glslc nor glslangValidator was found. Install a Vulkan "
      "shader compiler (Omarchy: sudo pacman -S shaderc glslang).")
endif()

function(omarchy_shader name source)
  set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
  set(spirv "${output_dir}/${name}.spv")
  set(header "${output_dir}/${name}.h")
  if(OMARCHY_GLSLC)
    add_custom_command(
      OUTPUT "${spirv}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
      COMMAND "${OMARCHY_GLSLC}" -O --target-env=vulkan1.3 ${ARGN}
              "${CMAKE_CURRENT_SOURCE_DIR}/${source}" -o "${spirv}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
      VERBATIM)
  else()
    add_custom_command(
      OUTPUT "${spirv}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
      COMMAND "${OMARCHY_GLSLANG}" -V -Os --target-env vulkan1.3 ${ARGN}
              "${CMAKE_CURRENT_SOURCE_DIR}/${source}" -o "${spirv}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
      VERBATIM)
  endif()
  add_custom_command(
    OUTPUT "${header}"
    COMMAND
      "${CMAKE_COMMAND}" -DINPUT=${spirv} -DOUTPUT=${header}
      -DNAME=${name} -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake"
    DEPENDS "${spirv}" "${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake"
    VERBATIM)
  set(OMARCHY_SHADER_HEADERS
      ${OMARCHY_SHADER_HEADERS} "${header}"
      PARENT_SCOPE)
endfunction()

omarchy_shader(elementwise_f32 shaders/elementwise.comp)
omarchy_shader(elementwise_f16 shaders/elementwise.comp -DUSE_FP16=1)
omarchy_shader(elementwise_bf16 shaders/elementwise.comp -DUSE_BF16=1)
omarchy_shader(cast_bool_f32 shaders/cast.comp -DSOURCE_BOOL=1)
omarchy_shader(cast_f16_f32 shaders/cast.comp -DSOURCE_FP16=1)
omarchy_shader(cast_f32_f16 shaders/cast.comp -DDEST_FP16=1)
omarchy_shader(cast_bf16_f32 shaders/cast.comp -DSOURCE_BF16=1)
omarchy_shader(cast_f32_bf16 shaders/cast.comp -DDEST_BF16=1)
omarchy_shader(cast_bf16_f16 shaders/cast.comp -DSOURCE_BF16=1 -DDEST_FP16=1)
omarchy_shader(cast_f16_bf16 shaders/cast.comp -DSOURCE_FP16=1 -DDEST_BF16=1)
omarchy_shader(cast_i32_f32 shaders/cast.comp -DSOURCE_I32=1)
omarchy_shader(cast_f32_i32 shaders/cast.comp -DDEST_I32=1)
omarchy_shader(cast_i32_f16 shaders/cast.comp -DSOURCE_I32=1 -DDEST_FP16=1)
omarchy_shader(cast_f16_i32 shaders/cast.comp -DSOURCE_FP16=1 -DDEST_I32=1)
omarchy_shader(cast_i32_bf16 shaders/cast.comp -DSOURCE_I32=1 -DDEST_BF16=1)
omarchy_shader(cast_bf16_i32 shaders/cast.comp -DSOURCE_BF16=1 -DDEST_I32=1)
omarchy_shader(cast_u32_f32 shaders/cast.comp -DSOURCE_U32=1)
omarchy_shader(reduce_f32 shaders/reduce_suffix.comp)
omarchy_shader(reduce_f16 shaders/reduce_suffix.comp -DUSE_FP16=1)
omarchy_shader(reduce_bf16 shaders/reduce_suffix.comp -DUSE_BF16=1)
omarchy_shader(matmul_f32 shaders/matmul.comp)
omarchy_shader(matmul_f16 shaders/matmul.comp -DUSE_FP16=1)
omarchy_shader(matmul_bf16 shaders/matmul.comp -DUSE_BF16=1)
omarchy_shader(fill_f32 shaders/fill.comp)
omarchy_shader(fill_f16 shaders/fill.comp -DUSE_FP16=1)
omarchy_shader(fill_bf16 shaders/fill.comp -DUSE_BF16=1)
omarchy_shader(softmax_f32 shaders/softmax_suffix.comp)
omarchy_shader(softmax_f16 shaders/softmax_suffix.comp -DUSE_FP16=1)
omarchy_shader(softmax_bf16 shaders/softmax_suffix.comp -DUSE_BF16=1)
omarchy_shader(logsumexp_f32 shaders/logsumexp_suffix.comp)
omarchy_shader(logsumexp_f16 shaders/logsumexp_suffix.comp -DUSE_FP16=1)
omarchy_shader(logsumexp_bf16 shaders/logsumexp_suffix.comp -DUSE_BF16=1)
omarchy_shader(select_f32 shaders/select.comp)
omarchy_shader(select_f16 shaders/select.comp -DUSE_FP16=1)
omarchy_shader(select_bf16 shaders/select.comp -DUSE_BF16=1)
omarchy_shader(compare_f32 shaders/compare.comp)
omarchy_shader(compare_f16 shaders/compare.comp -DUSE_FP16=1)
omarchy_shader(compare_bf16 shaders/compare.comp -DUSE_BF16=1)
omarchy_shader(compare_i32 shaders/compare.comp -DUSE_I32=1)
omarchy_shader(logical_or_bool shaders/logical_or.comp)
omarchy_shader(gather_f32 shaders/gather_rows.comp)
omarchy_shader(gather_f16 shaders/gather_rows.comp -DUSE_FP16=1)
omarchy_shader(gather_bf16 shaders/gather_rows.comp -DUSE_BF16=1)
omarchy_shader(gather_u32 shaders/gather_rows.comp -DUSE_U32=1)
omarchy_shader(copy_general_f32 shaders/copy_general.comp)
omarchy_shader(copy_general_f16 shaders/copy_general.comp -DUSE_FP16=1)
omarchy_shader(copy_general_bf16 shaders/copy_general.comp -DUSE_BF16=1)
omarchy_shader(argreduce_f32 shaders/argreduce_suffix.comp)
omarchy_shader(argreduce_f16 shaders/argreduce_suffix.comp -DUSE_FP16=1)
omarchy_shader(argreduce_bf16 shaders/argreduce_suffix.comp -DUSE_BF16=1)
omarchy_shader(arange_f32 shaders/arange.comp)
omarchy_shader(arange_f16 shaders/arange.comp -DUSE_FP16=1)
omarchy_shader(arange_bf16 shaders/arange.comp -DUSE_BF16=1)
omarchy_shader(arange_i32 shaders/arange.comp -DUSE_I32=1)
omarchy_shader(sort_f32 shaders/sort_suffix.comp)
omarchy_shader(sort_f16 shaders/sort_suffix.comp -DUSE_FP16=1)
omarchy_shader(sort_bf16 shaders/sort_suffix.comp -DUSE_BF16=1)
omarchy_shader(argsort_f32 shaders/sort_suffix.comp -DARGSORT=1)
omarchy_shader(argsort_f16 shaders/sort_suffix.comp -DUSE_FP16=1 -DARGSORT=1)
omarchy_shader(argsort_bf16 shaders/sort_suffix.comp -DUSE_BF16=1 -DARGSORT=1)
omarchy_shader(elementwise_i32 shaders/int_elementwise.comp)
omarchy_shader(scan_f32 shaders/scan_suffix.comp)
omarchy_shader(scan_f16 shaders/scan_suffix.comp -DUSE_FP16=1)
omarchy_shader(scan_bf16 shaders/scan_suffix.comp -DUSE_BF16=1)
omarchy_shader(searchsorted_f32 shaders/searchsorted_suffix.comp)
omarchy_shader(searchsorted_f16 shaders/searchsorted_suffix.comp -DUSE_FP16=1)
omarchy_shader(searchsorted_bf16 shaders/searchsorted_suffix.comp -DUSE_BF16=1)
omarchy_shader(searchsorted_i32 shaders/searchsorted_suffix.comp -DUSE_I32=1)
omarchy_shader(searchsorted_u32 shaders/searchsorted_suffix.comp -DUSE_U32=1)
omarchy_shader(random_bits_u32 shaders/random_bits.comp)
omarchy_shader(qmm_f32 shaders/qmm.comp)
omarchy_shader(qmm_f16 shaders/qmm.comp -DUSE_FP16=1)
omarchy_shader(qmm_bf16 shaders/qmm.comp -DUSE_BF16=1)
omarchy_shader(dequant_f32 shaders/dequant.comp)
omarchy_shader(dequant_f16 shaders/dequant.comp -DUSE_FP16=1)
add_custom_target(omarchy_shaders DEPENDS ${OMARCHY_SHADER_HEADERS})
add_dependencies(mlx omarchy_shaders)

target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/compute.cpp
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
          ${CMAKE_CURRENT_SOURCE_DIR}/ane/manifest.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ane/bundle.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/vulkan.cpp)
target_include_directories(mlx PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/shaders")

# The backend dlopens libvulkan at runtime; only headers are needed at compile
# time.
if(TARGET Vulkan::Headers)
  target_link_libraries(mlx PUBLIC Vulkan::Headers)
else()
  target_include_directories(mlx PUBLIC ${Vulkan_INCLUDE_DIRS})
endif()
