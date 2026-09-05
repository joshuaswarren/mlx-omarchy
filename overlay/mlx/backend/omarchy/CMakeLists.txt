# Env-gated GPU profiling harness (gpu_profiler.h). Compiled OUT by
# default: release builds carry zero profiling code. Turn on only for
# diagnostic builds (MLX_OMARCHY_GPU_PROFILE env var selects the output).
option(MLX_OMARCHY_GPU_PROFILING
       "Compile in the env-gated GPU profiling harness" OFF)
if(MLX_OMARCHY_GPU_PROFILING)
  target_compile_definitions(mlx PRIVATE MLX_OMARCHY_GPU_PROFILING)
endif()

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
omarchy_shader(select_i32 shaders/select.comp -DUSE_I32=1)
omarchy_shader(select_bool shaders/select.comp -DUSE_BOOL=1)
omarchy_shader(compare_f32 shaders/compare.comp)
omarchy_shader(compare_f16 shaders/compare.comp -DUSE_FP16=1)
omarchy_shader(compare_bf16 shaders/compare.comp -DUSE_BF16=1)
omarchy_shader(compare_i32 shaders/compare.comp -DUSE_I32=1)
omarchy_shader(logical_or_bool shaders/logical_or.comp)
omarchy_shader(compare_bool shaders/compare_bool.comp)
omarchy_shader(gather_f32 shaders/gather_rows.comp)
omarchy_shader(gather_f16 shaders/gather_rows.comp -DUSE_FP16=1)
omarchy_shader(gather_bf16 shaders/gather_rows.comp -DUSE_BF16=1)
omarchy_shader(gather_u32 shaders/gather_rows.comp -DUSE_U32=1)
omarchy_shader(copy_general_f32 shaders/copy_general.comp)
omarchy_shader(copy_general_f16 shaders/copy_general.comp -DUSE_FP16=1)
omarchy_shader(copy_general_bf16 shaders/copy_general.comp -DUSE_BF16=1)
omarchy_shader(copy_general_u32 shaders/copy_general.comp -DUSE_U32=1)
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
omarchy_shader(sort_i32 shaders/sort_suffix.comp -DUSE_I32=1)
omarchy_shader(sort_u32 shaders/sort_suffix.comp -DUSE_U32=1)
omarchy_shader(argsort_i32 shaders/sort_suffix.comp -DUSE_I32=1 -DARGSORT=1)
omarchy_shader(argsort_u32 shaders/sort_suffix.comp -DUSE_U32=1 -DARGSORT=1)
omarchy_shader(elementwise_i32 shaders/int_elementwise.comp)
omarchy_shader(elementwise_u32 shaders/int_elementwise.comp -DUSE_U32=1)
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
# FuseDecodeChains: interpreted elementwise-chain kernels (one dispatch
# per fusable tape chain; bf16 tapes are refused before chains run).
omarchy_shader(fused_chain_f32 shaders/fused_chain.comp)
omarchy_shader(fused_chain_f16 shaders/fused_chain.comp -DUSE_FP16=1)
# DecodeGemv: dedicated matrix-vector kernels for the decode shape
# (lhs single row); qmm_vec fuses affine dequantization into the dot,
# the same shader compiled with USE_SUBGROUP=1 produces the
# subgroup-reduction variant of the per-column sum (see PROTOCOL in
# tools/subgroup-bench/ for the rationale).
omarchy_shader(qmm_vec_f32 shaders/qmm_vec.comp)
omarchy_shader(qmm_vec_f16 shaders/qmm_vec.comp -DUSE_FP16=1)
omarchy_shader(qmm_vec_bf16 shaders/qmm_vec.comp -DUSE_BF16=1)
omarchy_shader(qmm_vec_subgroup_f32 shaders/qmm_vec.comp -DUSE_SUBGROUP=1)
omarchy_shader(qmm_vec_subgroup_f16 shaders/qmm_vec.comp -DUSE_FP16=1 -DUSE_SUBGROUP=1)
omarchy_shader(qmm_vec_subgroup_bf16 shaders/qmm_vec.comp -DUSE_BF16=1 -DUSE_SUBGROUP=1)
# DenseDecodeGemv: dense matrix-vector kernels for the decode shape
# (lhs single row), behind MLX_OMARCHY_DENSE_GEMV (default off) in
# primitives.cpp. Same source compiled with USE_SUBGROUP=1 produces the
# subgroup-reduction variant, dispatched only when the device reports
# subgroupSize == 32 and the ARITHMETIC feature bit.
omarchy_shader(matmul_vec_f32 shaders/matmul_vec.comp)
omarchy_shader(matmul_vec_f16 shaders/matmul_vec.comp -DUSE_FP16=1)
omarchy_shader(matmul_vec_bf16 shaders/matmul_vec.comp -DUSE_BF16=1)
omarchy_shader(matmul_vec_subgroup_f32 shaders/matmul_vec.comp -DUSE_SUBGROUP=1)
omarchy_shader(matmul_vec_subgroup_f16 shaders/matmul_vec.comp -DUSE_FP16=1 -DUSE_SUBGROUP=1)
omarchy_shader(matmul_vec_subgroup_bf16 shaders/matmul_vec.comp -DUSE_BF16=1 -DUSE_SUBGROUP=1)
# PrefillQmmTile: m-tiled quantized matmul for prefill (matrix_m > 1);
# one workgroup dequantizes each 16 x 16 weight block into shared f32
# once and applies it to a 16-row tile of x, amortizing the inline
# unpack the per-element qmm kernel repeats per output row. Dispatch is
# env-gated in primitives.cpp (MLX_OMARCHY_QMM_TILE).
omarchy_shader(qmm_tile_f32 shaders/qmm_tile.comp)
omarchy_shader(qmm_tile_f16 shaders/qmm_tile.comp -DUSE_FP16=1)
omarchy_shader(qmm_tile_bf16 shaders/qmm_tile.comp -DUSE_BF16=1)
omarchy_shader(dequant_f32 shaders/dequant.comp)
omarchy_shader(dequant_f16 shaders/dequant.comp -DUSE_FP16=1)
omarchy_shader(conv_f32 shaders/conv.comp)
omarchy_shader(conv_f16 shaders/conv.comp -DUSE_FP16=1)
omarchy_shader(conv_bf16 shaders/conv.comp -DUSE_BF16=1)

# Wave 5 (indexing and scatter). The u32 variants carry bitwise word
# storage, so float32, int32, and uint32 share them.
omarchy_shader(gather_axis_u32 shaders/gather_axis.comp -DUSE_U32=1)
omarchy_shader(gather_axis_f16 shaders/gather_axis.comp -DUSE_FP16=1)
omarchy_shader(gather_axis_bf16 shaders/gather_axis.comp -DUSE_BF16=1)
omarchy_shader(scatter_u32 shaders/scatter.comp -DUSE_U32=1)
omarchy_shader(scatter_f16 shaders/scatter.comp -DUSE_FP16=1)
omarchy_shader(scatter_bf16 shaders/scatter.comp -DUSE_BF16=1)
omarchy_shader(scatter_multi_u32 shaders/scatter.comp -DUSE_U32=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_multi_f16 shaders/scatter.comp -DUSE_FP16=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_multi_bf16 shaders/scatter.comp -DUSE_BF16=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_axis_u32 shaders/scatter_axis.comp -DUSE_U32=1)
omarchy_shader(scatter_axis_f16 shaders/scatter_axis.comp -DUSE_FP16=1)
omarchy_shader(scatter_axis_bf16 shaders/scatter_axis.comp -DUSE_BF16=1)
omarchy_shader(masked_scatter_u32 shaders/masked_scatter.comp -DUSE_U32=1)
omarchy_shader(masked_scatter_f16 shaders/masked_scatter.comp -DUSE_FP16=1)
omarchy_shader(masked_scatter_bf16 shaders/masked_scatter.comp -DUSE_BF16=1)
omarchy_shader(scatter_fadd_f32 shaders/scatter.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1)
omarchy_shader(scatter_fadd_f16 shaders/scatter.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_FP16=1)
omarchy_shader(scatter_fadd_bf16 shaders/scatter.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_BF16=1)
omarchy_shader(scatter_fadd_multi_f32 shaders/scatter.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_fcas_f32 shaders/scatter.comp -DUSE_FADD=1)
omarchy_shader(scatter_fcas_f16 shaders/scatter.comp -DUSE_FADD=1 -DUSE_FP16=1)
omarchy_shader(scatter_fcas_bf16 shaders/scatter.comp -DUSE_FADD=1 -DUSE_BF16=1)
omarchy_shader(scatter_fcas_multi_f32 shaders/scatter.comp -DUSE_FADD=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_bool shaders/scatter.comp -DUSE_BOOL=1)
omarchy_shader(scatter_bool_multi shaders/scatter.comp -DUSE_BOOL=1 -DUSE_MULTI_INDEX=1)
omarchy_shader(scatter_axis_fadd_f32 shaders/scatter_axis.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1)
omarchy_shader(scatter_axis_fadd_f16 shaders/scatter_axis.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_FP16=1)
omarchy_shader(scatter_axis_fadd_bf16 shaders/scatter_axis.comp -DUSE_FADD=1 -DUSE_HW_ATOMIC=1 -DUSE_BF16=1)
omarchy_shader(scatter_axis_fcas_f32 shaders/scatter_axis.comp -DUSE_FADD=1)
omarchy_shader(scatter_axis_fcas_f16 shaders/scatter_axis.comp -DUSE_FADD=1 -DUSE_FP16=1)
omarchy_shader(scatter_axis_fcas_bf16 shaders/scatter_axis.comp -DUSE_FADD=1 -DUSE_BF16=1)
omarchy_shader(scatter_axis_bool shaders/scatter_axis.comp -DUSE_BOOL=1)
omarchy_shader(clear_u32 shaders/clear_uint.comp)
omarchy_shader(reduce_general_f32 shaders/reduce_general.comp)
omarchy_shader(reduce_general_f16 shaders/reduce_general.comp -DUSE_FP16=1)
omarchy_shader(reduce_general_bf16 shaders/reduce_general.comp -DUSE_BF16=1)
omarchy_shader(reduce_general_i32 shaders/reduce_general.comp -DUSE_I32=1)
omarchy_shader(reduce_general_u32 shaders/reduce_general.comp -DUSE_U32=1)
omarchy_shader(anyall_f32 shaders/reduce_general.comp -DBOOL_OUT=1)
omarchy_shader(anyall_f16 shaders/reduce_general.comp -DBOOL_OUT=1 -DUSE_FP16=1)
omarchy_shader(anyall_bf16 shaders/reduce_general.comp -DBOOL_OUT=1 -DUSE_BF16=1)
omarchy_shader(anyall_i32 shaders/reduce_general.comp -DBOOL_OUT=1 -DUSE_I32=1)
omarchy_shader(anyall_u32 shaders/reduce_general.comp -DBOOL_OUT=1 -DUSE_U32=1)
omarchy_shader(anyall_bool shaders/reduce_general.comp -DBOOL_OUT=1 -DUSE_BOOL_INPUT=1)
omarchy_shader(scan_general_f32 shaders/scan_general.comp)
omarchy_shader(scan_general_f16 shaders/scan_general.comp -DUSE_FP16=1)
omarchy_shader(scan_general_bf16 shaders/scan_general.comp -DUSE_BF16=1)
omarchy_shader(scan_general_i32 shaders/scan_general.comp -DUSE_I32=1)
omarchy_shader(scan_general_u32 shaders/scan_general.comp -DUSE_U32=1)
omarchy_shader(hadamard_f32 shaders/hadamard.comp)
omarchy_shader(hadamard_f16 shaders/hadamard.comp -DUSE_FP16=1)
omarchy_shader(hadamard_bf16 shaders/hadamard.comp -DUSE_BF16=1)
omarchy_shader(block_mask_f32 shaders/block_mask.comp)
omarchy_shader(gather_mm_f32 shaders/gather_mm.comp)
omarchy_shader(gather_mm_f16 shaders/gather_mm.comp -DUSE_FP16=1)
omarchy_shader(gather_mm_bf16 shaders/gather_mm.comp -DUSE_BF16=1)
omarchy_shader(segmented_mm_f32 shaders/segmented_mm.comp)
omarchy_shader(segmented_mm_f16 shaders/segmented_mm.comp -DUSE_FP16=1)
omarchy_shader(segmented_mm_bf16 shaders/segmented_mm.comp -DUSE_BF16=1)
omarchy_shader(gather_qmm_f32 shaders/gather_qmm.comp)
omarchy_shader(gather_qmm_f16 shaders/gather_qmm.comp -DUSE_FP16=1)
omarchy_shader(gather_qmm_bf16 shaders/gather_qmm.comp -DUSE_BF16=1)
omarchy_shader(gather_qmm_nb_f32 shaders/gather_qmm.comp -DNO_BIAS=1)
omarchy_shader(gather_qmm_nb_f16 shaders/gather_qmm.comp -DUSE_FP16=1 -DNO_BIAS=1)
omarchy_shader(gather_qmm_nb_bf16 shaders/gather_qmm.comp -DUSE_BF16=1 -DNO_BIAS=1)
omarchy_shader(fft_f32 shaders/fft_c2c.comp)
omarchy_shader(fft_real_f32 shaders/fft_real_extract.comp)
omarchy_shader(fft_stage_f32 shaders/fft_stage.comp)
# Wave 7 (linear algebra). One workgroup per batch matrix, float32 only,
# matching the upstream CPU dtype contract.
omarchy_shader(linalg_cholesky_f32 shaders/linalg_cholesky.comp)
omarchy_shader(linalg_inverse_f32 shaders/linalg_inverse.comp)
omarchy_shader(linalg_lu_f32 shaders/linalg_lu.comp)
omarchy_shader(linalg_qr_f32 shaders/linalg_qr.comp)
omarchy_shader(linalg_eigh_f32 shaders/linalg_eigh.comp)
omarchy_shader(linalg_eig_f32 shaders/linalg_eig.comp)
omarchy_shader(linalg_svd_f32 shaders/linalg_svd.comp)
omarchy_shader(linalg_svd_finalize_f32 shaders/linalg_svd_finalize.comp)
omarchy_shader(fast_rms_norm_f32 shaders/fast_norm.comp)
omarchy_shader(fast_rms_norm_f16 shaders/fast_norm.comp -DUSE_FP16=1)
omarchy_shader(fast_rms_norm_bf16 shaders/fast_norm.comp -DUSE_BF16=1)
omarchy_shader(fast_layer_norm_f32 shaders/fast_norm.comp -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_f16 shaders/fast_norm.comp -DUSE_FP16=1 -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_bf16 shaders/fast_norm.comp -DUSE_BF16=1 -DLAYER_NORM=1)
omarchy_shader(fast_rms_norm_vjp_dx_f32 shaders/fast_norm_vjp_dx.comp)
omarchy_shader(fast_rms_norm_vjp_dx_f16 shaders/fast_norm_vjp_dx.comp -DUSE_FP16=1)
omarchy_shader(fast_rms_norm_vjp_dx_bf16 shaders/fast_norm_vjp_dx.comp -DUSE_BF16=1)
omarchy_shader(fast_layer_norm_vjp_dx_f32 shaders/fast_norm_vjp_dx.comp -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_vjp_dx_f16 shaders/fast_norm_vjp_dx.comp -DUSE_FP16=1 -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_vjp_dx_bf16 shaders/fast_norm_vjp_dx.comp -DUSE_BF16=1 -DLAYER_NORM=1)
omarchy_shader(fast_rms_norm_vjp_dw_f32 shaders/fast_norm_vjp_dw.comp)
omarchy_shader(fast_rms_norm_vjp_dw_f16 shaders/fast_norm_vjp_dw.comp -DUSE_FP16=1)
omarchy_shader(fast_rms_norm_vjp_dw_bf16 shaders/fast_norm_vjp_dw.comp -DUSE_BF16=1)
omarchy_shader(fast_layer_norm_vjp_dw_f32 shaders/fast_norm_vjp_dw.comp -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_vjp_dw_f16 shaders/fast_norm_vjp_dw.comp -DUSE_FP16=1 -DLAYER_NORM=1)
omarchy_shader(fast_layer_norm_vjp_dw_bf16 shaders/fast_norm_vjp_dw.comp -DUSE_BF16=1 -DLAYER_NORM=1)
omarchy_shader(fast_rms_norm_vjp_dw_reduce_f32 shaders/fast_norm_vjp_dw_reduce.comp)
omarchy_shader(fast_rms_norm_vjp_dw_reduce_f16 shaders/fast_norm_vjp_dw_reduce.comp -DUSE_FP16=1)
omarchy_shader(fast_rms_norm_vjp_dw_reduce_bf16 shaders/fast_norm_vjp_dw_reduce.comp -DUSE_BF16=1)
omarchy_shader(fast_cross_entropy_vjp_f32 shaders/fast_cross_entropy_vjp.comp)
omarchy_shader(fast_cross_entropy_vjp_f16 shaders/fast_cross_entropy_vjp.comp -DUSE_FP16=1)
omarchy_shader(fast_cross_entropy_vjp_bf16 shaders/fast_cross_entropy_vjp.comp -DUSE_BF16=1)
omarchy_shader(fast_cross_entropy_f32 shaders/fast_cross_entropy.comp)
omarchy_shader(fast_cross_entropy_f16 shaders/fast_cross_entropy.comp -DUSE_FP16=1)
omarchy_shader(fast_cross_entropy_bf16 shaders/fast_cross_entropy.comp -DUSE_BF16=1)
omarchy_shader(fast_rope_f32 shaders/fast_rope.comp)
omarchy_shader(fast_rope_f16 shaders/fast_rope.comp -DUSE_FP16=1)
omarchy_shader(fast_rope_bf16 shaders/fast_rope.comp -DUSE_BF16=1)
omarchy_shader(fast_rope_freqs_f32 shaders/fast_rope.comp -DUSE_FREQS=1)
omarchy_shader(fast_rope_freqs_f16 shaders/fast_rope.comp -DUSE_FREQS=1 -DUSE_FP16=1)
omarchy_shader(fast_rope_freqs_bf16 shaders/fast_rope.comp -DUSE_FREQS=1 -DUSE_BF16=1)
omarchy_shader(fp8_to_f32 shaders/fp8.comp -DFP8_TO_FLOAT=1)
omarchy_shader(fp8_to_f16 shaders/fp8.comp -DFP8_TO_FLOAT=1 -DUSE_FP16=1)
omarchy_shader(fp8_to_bf16 shaders/fp8.comp -DFP8_TO_FLOAT=1 -DUSE_BF16=1)
omarchy_shader(fp8_from_f32 shaders/fp8.comp)
omarchy_shader(fp8_from_f16 shaders/fp8.comp -DUSE_FP16=1)
omarchy_shader(fp8_from_bf16 shaders/fp8.comp -DUSE_BF16=1)
# Complex64 transport (Complex64Transport). One element is a vec2
# (re, im) pair of float32 words; no 16-bit storage features are
# involved. The cast matrix mirrors the upstream static_cast rules:
# any real source promotes to (x, 0), and a complex64 source reads
# real() for a real destination.
omarchy_shader(complex_elementwise shaders/complex_elementwise.comp)
omarchy_shader(complex_real shaders/complex_extract.comp)
omarchy_shader(complex_imag shaders/complex_extract.comp -DSELECT_IMAG=1)
omarchy_shader(cast_f32_c64 shaders/cast.comp -DDEST_C64=1)
omarchy_shader(cast_i32_c64 shaders/cast.comp -DSOURCE_I32=1 -DDEST_C64=1)
omarchy_shader(cast_u32_c64 shaders/cast.comp -DSOURCE_U32=1 -DDEST_C64=1)
omarchy_shader(cast_bool_c64 shaders/cast.comp -DSOURCE_BOOL=1 -DDEST_C64=1)
omarchy_shader(cast_f16_c64 shaders/cast.comp -DSOURCE_FP16=1 -DDEST_C64=1)
omarchy_shader(cast_bf16_c64 shaders/cast.comp -DSOURCE_BF16=1 -DDEST_C64=1)
omarchy_shader(cast_c64_f32 shaders/cast.comp -DSOURCE_C64=1)
omarchy_shader(fill_c64 shaders/fill.comp -DUSE_COMPLEX64=1)
omarchy_shader(copy_general_c64 shaders/copy_general.comp -DUSE_COMPLEX64=1)
add_custom_target(omarchy_shaders DEPENDS ${OMARCHY_SHADER_HEADERS})
add_dependencies(mlx omarchy_shaders)

target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
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
          ${CMAKE_CURRENT_SOURCE_DIR}/fused_chain.cpp
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
