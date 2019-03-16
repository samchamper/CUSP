cmake_minimum_required(VERSION 2.8.12.1)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

list(APPEND CMAKE_PREFIX_PATH "C:/OpenGL")
set(OPENGL_LIBRARY_DIR "C:/OpenGL/lib")
set(GLUT_INCLUDE_DIR "C:/OpenGL/include")
set(GLEW_INCLUDE_DIR "C:/OpenGL/include")
set(GLEW_LIBRARY_DIR "C:/OpenGL/lib")

PROJECT(cpu_gl_version)

add_executable(cpu_gl_version cpu_gl_version.cxx)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories(
    ${GLUT_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR} )
    
target_link_libraries(cpu_gl_version ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES} )

set_target_properties(cpu_gl_version PROPERTIES
                CXX_STANDARD 17 )

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT cpu_gl_version)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
