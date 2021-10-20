cmake_minimum_required(VERSION 3.10)
# cmake_minimum_required(VERSION 2.6)

# set the project name and version
project(la1 VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# Required for UCO Thinstation. May not be required in your own PC.
OPTION(THINSTATION	"Set to OFF at home" OFF)

IF(THINSTATION)
  SET(CMAKE_CXX_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=0 -m32")
  SET(XRAND Xrandr)
  link_directories(/usr/local/X11R7/lib)
  MESSAGE( STATUS "USING THINSTATION OPTIONS" )
ENDIF()

# add the executable
add_executable(la1 la1.cpp imc/MultilayerPerceptron.cpp)