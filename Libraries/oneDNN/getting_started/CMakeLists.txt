cmake_minimum_required(VERSION 3.4)
if("${CMAKE_CXX_COMPILER}" STREQUAL "")
    set(CMAKE_CC_COMPILER "icx")
    set(CMAKE_CXX_COMPILER "dpcpp")
endif()
project (getting_started)
file(COPY $ENV{DNNLROOT}/examples/ DESTINATION src FILES_MATCHING PATTERN "*.cpp")
file(COPY $ENV{DNNLROOT}/examples/CMakeLists.txt DESTINATION src)
file(COPY $ENV{DNNLROOT}/examples/dpcpp_driver_check.cmake DESTINATION src)
if(WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    file(COPY $ENV{DNNLROOT}/examples/template.vcxproj.user DESTINATION src)
endif()
add_subdirectory (${PROJECT_BINARY_DIR}/src bin)
