# Google MediaPipe Hand Tracking

This repo contains sample codes and applications using [Google MediaPipe](https://github.com/google/mediapipe) (python API) for realtime 3D hand tracking.

[**Blog**](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html) | [**Code**](https://google.github.io/mediapipe/solutions/hands.html) | [**Paper**](https://arxiv.org/abs/2006.10214) |  [**Video**](https://www.youtube.com/watch?v=I-UOrvxxXEk) | [**Model Card**](https://drive.google.com/file/d/1yiPfkhb4hSbXJZaSq9vDmhz24XVZmxpL/view)

![](doc/01_video.gif)

## Installation
The simplest way to run our implementation is to use anaconda.

You can create an anaconda environment called `mp` with
```
conda env create -f environment.yaml
conda activate mp
```

## Demo
* [00_image](code/00_image.py): Test with single image (Note: the display shows 2D keypoints, 2.5D depth and 3D view, the handedness classification may not be 100% accurate)
![](doc/00_image.png)

* [01_video](code/01_video.py): Test with video input (Note: it takes around 15 FPS on CPU)
![](doc/01_video.png)

* [02_gesture](code/02_gesture.py): Simple recognition of 11 gestures
![](doc/02_gesture.png)

* [03_game_rps](code/03_game_rps.py): Simple game of rock paper scissor
![](doc/03_game_rps.png)

* [04_hand_rom](code/04_hand_rom.py): Measuring hand range of motion (ROM)
![](doc/04_hand_rom.png)

## Limitations:
Estimating 3D hand pose from a single 2D image is an ill-posed problem and extremely challenging, thus the resulting hand ROM may not be accurate!
Please refer to the [model card](https://drive.google.com/file/d/1yiPfkhb4hSbXJZaSq9vDmhz24XVZmxpL/view) for more details on other types of limitations such as lighting, motion blur, etc.
