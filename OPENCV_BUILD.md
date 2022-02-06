# Obtaining opencv-python with CUDA support

1. Follow steps 1 and 2 in **Manual Build** at [OpenCV-Python Package](https://pypi.org/project/opencv-python/#:~:text=the%20CI%20environment.-,Manual%20builds,delocate%20(same%20as%20auditwheel%20but%20for%20macOS)%20for%20better%20portability,-Manual%20debug%20builds);
2. [Discover your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus#compute). You will need to replace in the flag `-DCUDA_ARCH_BIN=<YOUR CUDA VERSION>` on step 3;
3. Install dependencies:
   - [You MUST have CUDA installed on your system](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
   - Install the required libraries that are in the [Alternative Method](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
4. Build with this command, this may take a while:
```
ENABLE_HEADLESS=1 ENABLE_CONTRIB=1 CMAKE_ARGS="-DWITH_TBB=ON -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_CUBLAS=ON -DWITH_FFMPEG=ON -DWITH_GSTREAMER=ON -DWITH_OPENJPEG=ON -DWITH_JPEG=ON -DOPENCV_ENABLE_NONFREE=ON -DWITH_OPENMP=ON -DWITH_LAPACK=ON -DCUDA_FAST_MATH=1 -DENABLE_FAST_MATH=1 -DOPENCV_DNN_CUDA=ON -DWITH_V4L=ON -DWITH_OPENGL=ON -DWITH_GSTREAMER=ON -DINSTALL_PYTHON_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF -DCUDA_ARCH_BIN=7.5" pip wheel . --verbose
```
5. The compiled .whl files will be in the the opencv-python, you can install them with: `pip install <fileneame>.whl`