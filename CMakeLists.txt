cmake_minimum_required(VERSION 2.8.12.1)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION OFF)
list(APPEND CUDA_NVCC_FLAGS "-gencode;-O2;-DVERBOSE")

PROJECT(CUSP LANGUAGES CUDA CXX)

#find_package(CUDA REQUIRED) 
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
#find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
add_executable(cpu_version cpu_version.cxx)
CUDA_ADD_EXECUTABLE(CUSP CUSP.cu)

#find_library( CUDA_LIBRARY cuda ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
#find_library( CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
#find_library( CUDA_NVRTC_LIBRARY nvrtc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )

#target_include_directories(CUSP PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(VTK_LIBRARIES)
  target_link_libraries(cpu_version ${VTK_LIBRARIES})
  target_link_libraries(CUSP ${VTK_LIBRARIES}})
else()
  target_link_libraries(cpu_version vtkHybrid)
  target_link_libraries(CUSP vtkHybrid)
endif()

set(CUSP_LINK_LIBRARIES
    ${CUSP_LINK_LIBRARIES}
    ${CUDA_NVRTC_LIBRARY} ${CUDA_LIBRARY} ${CUDART_LIBRARY}
    CACHE INTERNAL "" FORCE )

set_target_properties(cpu_version PROPERTIES
                CXX_STANDARD 17
                CUDA_STANDARD 14
                CXX_EXTENSIONS OFF )
set_target_properties(CUSP PROPERTIES
                CXX_STANDARD 17
                CUDA_STANDARD 14
                CXX_EXTENSIONS OFF )

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT CUSP)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
