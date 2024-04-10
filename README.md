# GO PRO HERO 10

This repository contains source code that enables communication with the GoPro Hero 10 Black camera through HTTP requests and provides functionality to query and control various camera parameters. Additionally, it includes a class that utilizes OpenCV for live previewing of the camera feed and allows for manipulation of visualization parameters.

<br>

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
- [GoPro Hero 10 Black](#gopro-hero-10-black)
   * [Battery Specs](#battery-specs)
   * [Video Specs](#video-specs)
      + [Aspect Ratio](#aspect-ratio)
      + [Digital Lenses](#digital-lenses)
      + [Resolutions](#resolutions)
  * [Webcam Constraints](#webcam-constraints)
  * [Preview Stream](#preview-stream)
- [Webcam Sequence Diagram](#webcam-sequence-diagram)
   * [Elements:](#elements)
   * [Diagram](#diagram)
- [Useful Links](#useful-links)

<br>

## Features

- **GoPro Hero 10 Webcam Control Class** [`goProHero10.py`](https://github.com/Gugon11/goProHero10/blob/main/src/Camera/goProHero10.py): implements HTTP communication with the GoPro Hero 10 Black camera, enabling retrieval of camera values and parameter adjustment.

- **OpenCV Live Feed Preview Class** [`camera.py`](https://github.com/Gugon11/goProHero10/blob/main/src/Camera/camera.py): utilizes OpenCV to provide real-time previewing of the webcam feed. Users can explore techniques such as edge detection, image segmentation, and others according to specific tasks, requirements, or desired results.


<br>

## Getting Started

To get started with using the classes provided in this repository, follow these steps:

- Clone the repository to your local machine:
```bash
    git clone https://github.com/Gugon11/goProHero10.git
```

- Ensure you have the necessary dependencies installed. This may include Python, OpenCV `pip install opencv-python`, and any additional libraries required for HTTP communication with the GoPro camera.

- Explore the provided classes and example usage within the repository to understand how to integrate them into your projects.


<br>



## GoPro Hero 10 Black

- Product manual: [HERO 10 Black](https://gopro.com/content/dam/help/hero10-black/manuals/HERO10Black_UM_ENG_REVB.pdf) (video/photo specs starting page nr. 63/77)

<br>

### Battery Specs

```
RECORDING WHEN PLUGGED INTO A POWER SOURCE
You can use the USB-C cable that came with your camera to shoot videos and photos while your camera is plugged into a USB-charging adapter, the GoPro Supercharger, or other external power sources.

This is perfect for capturing long videos and time lapse events.

Be careful not to touch the area near the SD card slot when recording.

Disturbing the SD card could interrupt or stop the recording. Your camera will let you know if this happens. Your content will not be damaged or lost due to this disturbance, but you may need to restart your camera to continue.

Even though your camera is charging, the battery will not charge during recording. It will start charging when you stop recording.

You cannot record while your camera is plugged into a computer.
```

Thus, when using the GoPro as a webcam, the battery drains faster than it can charge while plugged into 3.0 USB.

<br>

### Video Specs

#### Aspect Ratio

![imagem](https://github.com/conradoguimaraes/goProHero10/assets/98216516/f60bf874-a226-4ecd-a378-bea2ddfc92c9)

<br>

#### Digital Lenses

![imagem](https://github.com/conradoguimaraes/goProHero10/assets/98216516/effc646d-e186-48bc-8b54-f5e00b8fe1b9)

<br>

#### Resolutions

![image01](https://github.com/conradoguimaraes/goProHero10/assets/98216516/ab507b64-e130-4d6f-a973-623c7afe3118)

```
Shooting high-resolution or high-fps video when it’s warm out can cause your camera to heat up and use more power.
```




### Webcam Constraints

From [GoPro Support](https://community.gopro.com/s/article/GoPro-Webcam?language=en_US):
```
Webcam Mode resolution limitations: "Here you can choose between [1080p] (default) or [720p]."

    Webcam Resolution
    ID 	Resolution 	Supported Cameras
    4 	480p 	Hero 10 Black, Hero 9 Black
    7 	720p 	Hero 12 Black, Hero 9 Black, Hero 10 Black, Hero 11 Black
    12 	1080p 	Hero 12 Black, Hero 9 Black, Hero 10 Black, Hero 11 Black

    Webcam Field-of-View
    ID 	FOV 	Supported Cameras
    0 	Wide 	Hero 12 Black, Hero 9 Black, Hero 10 Black, Hero 11 Black
    2 	Narrow 	Hero 12 Black, Hero 9 Black, Hero 10 Black, Hero 11 Black
    3 	Superview 	Hero 12 Black, Hero 9 Black, Hero 10 Black, Hero 11 Black
    4 	Linear 	Hero 12 Black, Hero 9 Black, Hero 10 Black, Hero 11 Black
```
### Preview Stream

OBS: **Preview Stream** is not supported on GoPro Hero 10

```
When the preview stream is started, the camera starts up a UDP client and begins writing MPEG Transport Stream data to the client on port 8554. In order to stream this data, the client must implement a UDP connection that binds to the same port and decode the data.

**Start Preview Stream**

Supported Protocols:
USB
WIFI

query Parameters
port, integer
Example: port=8556

Port to use for Preview Stream. Defaults to 8554 if not set

Not supported on:
    Hero 11 Black Mini
    Hero 11 Black
    **Hero 10 Black**
    Hero 9 Black


```




<br>

<br>

<br>

## Webcam Sequence Diagram

### Elements:
- **main** [`main.py`](https://github.com/conradoguimaraes/goProHero10/blob/main/src/Camera/main.py): main script
- **goProHero10 API** [`goProHero10.py`](https://github.com/conradoguimaraes/goProHero10/blob/main/src/Camera/goProHero10.py): _goProHero10 class_ that handles the HTTP communication with the GoPro Hero 10 Camera (Hardware)
- **camera** [`camera.py`](https://github.com/conradoguimaraes/goProHero10/blob/main/src/Camera/camera.py): _class_ that implements and handles, through _openCV_, the live feed, keypress events, image processing, etc;
- **GoPro Hero 10**: hardware

### Diagram

```mermaid
sequenceDiagram
    main->>main: Set desired Resolution/FOV
    main->>camera: initialize class object
    main->>goProHero10 API: initialize class object
    loop while failure exists
      main->>goProHero10 API: Change {resolution,fov}
      goProHero10 API->>GoPro Hero 10: HTTP Request: startWebcam{resolution,fov}
      GoPro Hero 10->>GoPro Hero 10: Try to Adjust Lens
      GoPro Hero 10->>goProHero10 API: HTTP Response
      goProHero10 API->>main: Return Code
      alt failure was detected
          main->>goProHero10 API: HTTP Request: exitWebcam
          goProHero10 API->>GoPro Hero 10: turn off Webcam
          GoPro Hero 10->>goProHero10 API: HTTP Response
          goProHero10 API->>main: Return Code
      end
   end

   main->>camera: start live feed
   camera->>GoPro Hero 10: start USB communication
   GoPro Hero 10->>camera: camera is available and ready

   loop while live feed is available
     camera->>GoPro Hero 10: capture image
     GoPro Hero 10->>camera: return captured image
     camera->>main: image preview
   end
    
```


## Useful Links

- Open GoPro HTTP API: [OpenGoPro/HTTP](https://gopro.github.io/OpenGoPro/http)
- GoPro Support webpage: [GoPro-Webcam](https://community.gopro.com/s/article/GoPro-Webcam?language=en_US)

