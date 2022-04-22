# Distance estimation with CARLA, YOLO, IPM and Stereoscopy

This project implements a realtime YOLO sensor into CARLA and adds distance estimation to it with IPM and two stereoscopy methods. This project is based on CARLA code examples. Developed on Ubuntu 20.04 running on WSL2 + WSLg on Windows 11.

## Requirements

- Python 3.8.10 (other versions might work, but not tested)
- [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [OpenCV 4.5.5 compiled with CUDA/DNN](/OPENCV_BUILD.md) support
    - You may use OpenCV without CUDA, but performance will be very poor
    - [Alternative method](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
- [CARLA 0.9.12](https://github.com/carla-simulator/carla/releases/tag/0.9.12)
- YOLO models:
    - [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- From [Darknet repository](https://github.com/pjreddie/darknet):
    - yolov3.cfg
    - cocos.names
- From [Pytorch-YoloV4](https://github.com/WildflowerSchools/pytorch-YOLOv4/)
    - [yolov4-weigths](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)
    - yolov4.cfg

## Installation

- Copy yolov3.weights, yolov4.weights, yolov3.cfg, yolov4.cfg and cocos.names to the `models` folder.
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
- **2**: YoloV3 Sensor view
- **3**: YoloV4 Sensor view
- **4**: YoloV5 Sensor view
- **5**: IPM Sensor view
- **6**: YoloV3 Stereo Sensor view
- **7**: YoloV4 Stereo Sensor view
- **8**: YoloV5 Stereo Sensor view
- **TAB**: Change camera
- **P**: Start autopilot (if supported on map)
- **u**: Start dataset generation
- *Other controls*: check the terminal output for default CARLA controls

After starting the application you should see a window showing a car and the loaded map. You can manually control this vehicle or press **P** to use the autopilot. For the intended usage of this project it is recommended to use the autopilot, the Yolo Sensor View (press **1**) and the dashcam view.

## Video 

[![Demo on Youtube](https://i3.ytimg.com/vi/YahcyJkUCWA/hqdefault.jpg)](https://www.youtube.com/watch?v=YahcyJkUCWA)
> This video is on an older version, a new video comming soon

## Know Issues and Limitations

* Sometimes the YoloV3 and YoloV4 sensors will hang with a CUDA error
* Distance estimation only shows correct data on the initial camera position
* Changing Yolo versions during runtime degrade performance
* Dataset generation is unreliable, especially on sync mode. To generate a dataset use ```run-async```