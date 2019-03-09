cmake_minimum_required(VERSION 2.8.12.1)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


PROJECT(cpu_version)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
add_executable(cpu_version cpu_version.cxx)


if(VTK_LIBRARIES)
  target_link_libraries(cpu_version ${VTK_LIBRARIES})
else()
  target_link_libraries(cpu_version vtkHybrid)
endif()

set_target_properties(cpu_version PROPERTIES
                CXX_STANDARD 17 )

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT cpu_version)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
