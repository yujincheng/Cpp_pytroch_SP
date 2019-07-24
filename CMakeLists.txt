cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

set(Torch_DIR /home/tsui/yujc/testcpptorch/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED )
include_directories(include)

add_executable(example-app src/example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_link_libraries(example-app "${OpenCV_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
