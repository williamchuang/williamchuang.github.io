---
layout: post
title: "CS420 Game Engineering Project 3"
date: 2016-06-24
---

Racing Cubes--Humans VS. Self-Driving Cubes on A Möbius Strip<br>
The main project was growing from my Tank project. In order to reach a better performance, for the Neural Nets, I started from scratch, and didn't use any NN/ML libraries (I rewrote some of them (based on the source code of Numpy, TensorFlow, and some recent published notions of deep neural nets), then implemented in this project) so that this NN code could be the best fit of this project. Thus far, I couldn't find anyone did this in Ogre3D, so if you know there are any instances that are implementing Neural Nets in Ogre3D that are using Ogre's build-in camera as their NN's inputs like my project did, please let me know (I would like to learn how other people solve the same problems that were met and solved in this project).<br>

<iframe width="480" height="360" src="https://www.youtube.com/embed/bRahjN1olgk" frameborder="0" allowfullscreen></iframe><br>

The goal for Supervised Learning AI cubes/cars is to hit as many coins as possible.<br>
Before training, the initial conditions of NN AIs were using a set of random numbers from a normal Gaussian distribution between zero and one form the weighting matrices, so the output for the control (go forward, turn left, turn right, and reverse) would be randomly chosen form one of them. After 2-3 epochs of training, the accuracy could end up with 96% (additionally, in humans' result, the accuracy is ~80%). The position of all coins is using the same Gaussian distribution for generating a random position within the range of the parameters (u, v) that parameterize the Möbius Strip. All the meshes, including the Möbius Strip, are built by using Blender 3d.<br>

