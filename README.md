# Google MediaPipe

This repo contains sample codes and applications using [Google MediaPipe](https://github.com/google/mediapipe) (python API) for realtime 3D hand tracking and 2D upper body pose estimation.

## Hand Tracking
[**Blog**](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html) | [**Code**](https://google.github.io/mediapipe/solutions/hands.html) | [**Paper**](https://arxiv.org/abs/2006.10214) |  [**Video**](https://www.youtube.com/watch?v=I-UOrvxxXEk) | [**Model Card**](https://drive.google.com/file/d/1yiPfkhb4hSbXJZaSq9vDmhz24XVZmxpL/view)

## Body Tracking
[**Blog**](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html) | [**Code**](https://google.github.io/mediapipe/solutions/pose) | [**Paper**](https://arxiv.org/abs/2006.10204) |  [**Video**](https://www.youtube.com/watch?v=YPpUOTRn5tA&feature=emb_logo) | [**Model Card**](https://drive.google.com/file/d/1zhYyUXhQrb_Gp0lKUFv1ADT3OCxGEQHS/view)

![](doc/01_video.gif)

Click on the image below to view the video on YouTube:

[![](https://img.youtube.com/vi/pxVj8oB-g-w/0.jpg)](https://www.youtube.com/watch?v=pxVj8oB-g-w)


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

* [05_hand_body](code/05_hand_body.py): Test with single image of upper body to detect both upper body and hand joints (Note: the image for subject with body marker is adapted from [An Asian-centric human movement database capturing activities of daily living](https://www.nature.com/articles/s41597-020-00627-7?sf237508323=1) and the image of Mona Lisa is adapted from [Wiki](https://upload.wikimedia.org/wikipedia/commons/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg))
![](doc/05_hand_body.png)

* [06_hand_body_video](code/06_hand_body_video.py): Test with video input to detect both upper body and hand joints (Note: the video is adapted from [Fugl-Meyer Assessment of Motor Recovery after Stroke](https://www.youtube.com/watch?v=B70qDfl3LyA&gl=SG))
Click on the image below to view the video on YouTube:

[![](https://img.youtube.com/vi/pxVj8oB-g-w/0.jpg)](https://www.youtube.com/watch?v=pxVj8oB-g-w)

## Limitations:
Estimating 3D hand pose from a single 2D image is an ill-posed problem and extremely challenging, thus the resulting hand ROM may not be accurate!
Please refer to the [model card](https://drive.google.com/file/d/1yiPfkhb4hSbXJZaSq9vDmhz24XVZmxpL/view) for more details on other types of limitations such as lighting, motion blur, etc.
