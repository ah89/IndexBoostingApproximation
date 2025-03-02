cmake_minimum_required(VERSION 3.10)
project(Sig2Model)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories for your project
include_directories(include)

# Find Google Test (ensure it is installed or available as a submodule)
find_package(GTest REQUIRED)

# Add your library (sig2model)
add_library(sig2model
    src/sig2model.cpp
    src/radix_spline.cpp
    src/neural_network.cpp
    src/gmm.cpp
    src/buffer_manager.cpp
    src/control_unit.cpp
    src/placeholder_strategy.cpp
)

# Add the test executable
add_executable(sig2model_test
    tests/test_sig2model.cpp
)

# Link Google Test libraries dynamically
target_link_libraries(sig2model_test PRIVATE GTest::GTest GTest::Main)

# Link your project library to the test executable
target_link_libraries(sig2model_test PRIVATE sig2model)

# Enable testing
enable_testing()
add_test(NAME sig2model_test COMMAND sig2model_test)