# CARLA + YOLO Realtime Integration

This project implements a realtime YOLO sensor into CARLA, using OpenCV. This project is based on CARLA code examples. Developed on Ubuntu 20.04 running on WSL2 + WSLg on Windows 11.

## Requirements

- Python 3.8.10 (other versions might work, but not tested)
- [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [OpenCV 4.5.5 compiled with CUDA/DNN](/OPENCV_BUILD.md) support
    - You may use OpenCV without CUDA, but performance will be very poor
    - [Alternative method](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
- [CARLA 0.9.12](https://github.com/carla-simulator/carla/releases/tag/0.9.12)
- YOLOv3 model:
    - [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- From [Darknet repository](https://github.com/pjreddie/darknet):
    - yolov3.cfg
    - cocos.names

## Installation

- Copy yolov3.weights, yolov3.cfg and cocos.names to the `models` folder.
- Execute `pip install -r requirements.txt`

## Execution

- On Windows with WSL:
    - Run CARLA server on Windows Host
    - Execute `make run`. It should automatically get the Host IP address and start the GUI application

- On Linux or to connect to a localhost CARLA server (not tested):
  - Start CARLA server
  - Execute `make run-linux`

## Usage

Keyboard shortcuts:
- **1**: Normal camera view
- **2**: Yolo Sensor view
- **3**: Yolo Sensor view with lens distortion
- **4**: Yolo Sensor with Full HD base image
- **TAB**: Change view
- **P**: Start autopilot
- *Other controls*: check the terminal output for default CARLA controls

After starting the application you should see a window showing a car and the loaded map. You can manually control this vehicle or press **P** to use the autopilot. For the intended usage of this project it is recommended to use the autopilot, the Yolo Sensor View (press **1**) and the dashcam view (press **TAB** once).