if(NOT INPUT OR NOT OUTPUT OR NOT NAME)
  message(FATAL_ERROR "INPUT, OUTPUT, and NAME are required")
endif()

if(NAME MATCHES "^fast_(rms_norm|rope|rope_freqs)_(f32|f16|bf16)$")
  execute_process(COMMAND python3 "${CMAKE_CURRENT_LIST_DIR}/preserve-rms-fma.py" "${INPUT}"
                  COMMAND_ERROR_IS_FATAL ANY)
endif()

file(READ "${INPUT}" bytes HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${bytes}")
file(
  WRITE "${OUTPUT}"
  "#pragma once\n\n#include <cstddef>\n\nnamespace mlx::core::omarchy::shaders {\n\nalignas(4) inline constexpr unsigned char ${NAME}[] = {${bytes}};\ninline constexpr size_t ${NAME}_size = sizeof(${NAME});\n\n} // namespace mlx::core::omarchy::shaders\n")
