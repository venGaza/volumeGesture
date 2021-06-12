### Volume Gesture

#### Table of Contents
- [Table of Contents](#table-of-contents)
- [Project Overview](#project-overview)
- [Demo](#demo)
- [Project Details](#project-details)
  - [Language](#language)
  - [Primary Modules](#primary-modules)
  - [Design Thoughts](#design-thoughts)
  - [Potential Use Cases](#potential-use-cases)

#### Project Overview

Computer vision is a scientific field that allows the machine to develop an understanding of an image or video. With the use of openCV and Google's MediaPipe library, you can easily integrate computer vision capabilities into various projects. To demonstrate this idea, I built this project to demonstrate how the volume on the computer could be controlled by gesturing with the thumb and index fingers.

#### Demo

<p align="center"><img src="../../images/portfolio/gifs/volumeControl.gif?raw=true" width="70%" height="70%"></p>
  
#### Project Details

##### Language 
- Python

##### Primary Modules
- openCV
- MediaPipe

##### Design Thoughts
- MediaPipe is blazingly fast even running on just the CPU. 
- This was a straightforward project and most of the time was spent on tuning the sensitivity of the volume control. Due to this sensitivity, I realized how much the z-value (the distance of the hand from the webcam) impacted the control. When the hand is far away from the camera, the volume can only be adjusted between a small range. Conversely, when the hand is very close to the camera, the volume tends to hit the maximum immediately. A potential solution would be figure out a way to estimate the z-value and interpolate this value with the distance between the thumb and index fingers.
- Sometimes the volume does not stay at the desired volume. This is due to the program still taking distance measurements as the user is moving their hand away from the webcam field of view. I worked around this issue by increasing the hand detection confidence of the neural net so that if the user moves their hand fast enough, the system will not detect it. You do have to keep in mind that boosting the confidence too high though will negatively impact the performance of the volume control as the net is more concerned with increasing its precision.
- Using the oshascript system call was a very expensive operation and dropped my framerate down to the single digits. 
  
##### Potential Use Cases
- Controlling advertisement mediums in retail locations such as malls
- Controlling screens in hospitals where contamination could pose a risk