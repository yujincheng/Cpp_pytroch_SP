# Cpp_pytroch_SP

## down load c++ pytroch
```
wget https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-latest.zip

unzip libtorch-shared-with-deps-latest.zip
```

## indicate Torch in CMakeLists.txt

The 4-5 lines in the CMakeLists.txt

```
set(Torch_DIR /home/tsui/yujc/testcpptorch/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
```

Change the Torch\_DIR to your path

## build the project

```
mkdir build
cmake ..
./example-app
```

Then the keypoints is drawn on opencv. The descriptors are printed on cmd lines.

## Profiling

The data transfer from CUDA to CPU is very time consuming.