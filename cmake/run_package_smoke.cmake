if(NOT DEFINED project_build_dir)
  message(FATAL_ERROR "project_build_dir is required")
endif()
if(NOT DEFINED source_dir)
  message(FATAL_ERROR "source_dir is required")
endif()
if(NOT DEFINED binary_dir)
  message(FATAL_ERROR "binary_dir is required")
endif()
if(NOT DEFINED install_dir)
  message(FATAL_ERROR "install_dir is required")
endif()
if(NOT DEFINED ctest_command)
  message(FATAL_ERROR "ctest_command is required")
endif()

file(REMOVE_RECURSE "${binary_dir}" "${install_dir}")
file(MAKE_DIRECTORY "${binary_dir}" "${install_dir}")

execute_process(
  COMMAND ${CMAKE_COMMAND} --install "${project_build_dir}" --prefix "${install_dir}"
  RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR "Failed to install cuflash from ${project_build_dir}")
endif()

set(configure_args
  -S "${source_dir}"
  -B "${binary_dir}"
  -DCMAKE_PREFIX_PATH=${install_dir}
)
if(DEFINED generator AND NOT generator STREQUAL "")
  list(APPEND configure_args -G "${generator}")
endif()
if(DEFINED build_type AND NOT build_type STREQUAL "")
  list(APPEND configure_args -DCMAKE_BUILD_TYPE=${build_type})
endif()
if(DEFINED c_compiler AND NOT c_compiler STREQUAL "")
  list(APPEND configure_args -DCMAKE_C_COMPILER=${c_compiler})
endif()
if(DEFINED cxx_compiler AND NOT cxx_compiler STREQUAL "")
  list(APPEND configure_args -DCMAKE_CXX_COMPILER=${cxx_compiler})
endif()
if(DEFINED cuda_architectures AND NOT cuda_architectures STREQUAL "")
  list(APPEND configure_args -DCMAKE_CUDA_ARCHITECTURES=${cuda_architectures})
endif()
if(DEFINED package_smoke_enable_asan)
  list(APPEND configure_args
       -DCUFLASH_PACKAGE_SMOKE_ENABLE_ASAN=${package_smoke_enable_asan})
endif()
if(DEFINED package_smoke_enable_ubsan)
  list(APPEND configure_args
       -DCUFLASH_PACKAGE_SMOKE_ENABLE_UBSAN=${package_smoke_enable_ubsan})
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} ${configure_args}
  RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR "Failed to configure downstream package smoke project")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} --build "${binary_dir}"
  RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "Failed to build downstream package smoke project")
endif()

execute_process(
  COMMAND ${ctest_command} --test-dir "${binary_dir}" --output-on-failure
  RESULT_VARIABLE test_result
)
if(NOT test_result EQUAL 0)
  message(FATAL_ERROR "Downstream package smoke tests failed")
endif()
