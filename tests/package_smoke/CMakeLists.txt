cmake_minimum_required(VERSION 3.18)
project(cuflash_attn_package_smoke LANGUAGES CXX)

option(CUFLASH_PACKAGE_SMOKE_ENABLE_ASAN "Link the smoke consumer with AddressSanitizer" OFF)
option(CUFLASH_PACKAGE_SMOKE_ENABLE_UBSAN
       "Link the smoke consumer with UndefinedBehaviorSanitizer" OFF)

find_package(cuflash_attn REQUIRED CONFIG)

add_executable(cuflash_attn_package_smoke main.cpp)
target_link_libraries(cuflash_attn_package_smoke PRIVATE cuflash::cuflash_attn)

if(CUFLASH_PACKAGE_SMOKE_ENABLE_ASAN OR CUFLASH_PACKAGE_SMOKE_ENABLE_UBSAN)
    set(package_smoke_sanitizer_flags "")
    if(CUFLASH_PACKAGE_SMOKE_ENABLE_ASAN)
        list(APPEND package_smoke_sanitizer_flags "-fsanitize=address")
    endif()
    if(CUFLASH_PACKAGE_SMOKE_ENABLE_UBSAN)
        list(APPEND package_smoke_sanitizer_flags "-fsanitize=undefined")
    endif()
    target_compile_options(cuflash_attn_package_smoke PRIVATE
                           ${package_smoke_sanitizer_flags} -fno-omit-frame-pointer)
    target_link_options(cuflash_attn_package_smoke PRIVATE ${package_smoke_sanitizer_flags})
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cuflash_attn_package_smoke COMMAND cuflash_attn_package_smoke)
endif()
