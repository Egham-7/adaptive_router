# =============================================================================
# Adaptive Router Core - Shared Compile Options
# =============================================================================

function(adaptive_target_compile_options target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -Wpedantic
        -Wconversion -Wsign-conversion
        -Wnon-virtual-dtor -Woverloaded-virtual
        -Wold-style-cast -Wcast-qual
        -Wformat=2 -Wimplicit-fallthrough
      >
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${ADAPTIVE_WARNINGS_AS_ERRORS}>>:-Werror>
    )

    if(ADAPTIVE_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
      target_compile_options(${target} PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer
      )
      target_link_options(${target} PRIVATE
        -fsanitize=address,undefined
      )
    endif()

  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${target} PRIVATE
      /W4 /permissive- /Zc:__cplusplus
      $<$<BOOL:${ADAPTIVE_WARNINGS_AS_ERRORS}>:/WX>
    )
  endif()
endfunction()