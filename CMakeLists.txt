cmake_minimum_required(VERSION 2.6)
project(Foo)

# find library dependencies (comment if there are )
FIND_PACKAGE( CasADi REQUIRED )

# SET(DEP_INCLUDE_DIRS 
#       ${DUMMYDEP_INCLUDE_DIR})

SET(DEP_LIBS 
      casadi)

# Set variables
include(${CMAKE_SOURCE_DIR}/cmake/SetEnv.cmake)

# Library sources
add_subdirectory(${LIBRARY_FOLDER})

# Library examples
add_subdirectory(example_internal)

# Install targets
include(${CMAKE_SOURCE_DIR}/cmake/InstallConfig.cmake)
