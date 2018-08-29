<p align="center">
  <img src="imgs/logo-color.png" height=350/>
</p>

### Problem

It is often difficult to be able to run inference on trained models of open source projects. This is mainly because most deep learning repos are all about describing new architectures, explaining training processes and/or publishing amazing metric results. The focus is more on the **model description and explanation rather than on the model use**.

When we want to run some model, we often encounter complex installation steps or even model unavailability (train it yourself!).

### Goal

The goal of this repo is to provide a place where we can **make use of the trained models**. Each model will be hosted in it's own site where a README will guide the user over simple and easy steps on how to run the model (often involving an auxiliary class also provided in the site). All models will be also hosted on this repo, under the `releases` section.

### Contribute! :raised_hands:

The only way to grow this collection is with your help. **If you know how to run a traditional model and/or you built one and you wish to share it, you're welcome**. Read [contribute.md](contribute.md).

### Note

Python only for now! :snake:

## How to use

> 1. Pick a **category**.
> 2. Pick the **task**
> 3. Pick the **model** and go to the model site!

# Categories:

This index will take you to all models of the category, regardless framework

- [Computer Vision](#computer-vision-tasks)
- [Natural Language Processing](#natural-language-processing-tasks)
- [Audio](#audio-tasks)
- [Other](#other)

# Computer Vision tasks

- [Image Classification](#image-classification)
- [Object Detection](#object-detection)
- [Pose Estimation](#pose-estimation)
- [Face Detection](#face-detection)
- [Instance Segmentation](#instance-segmentation)
- [Image Enhancement](#image-enhancement)
- [Image Captioning](#image-captioning)
- [Other](#other-computer-vision-models)

# Natural Language Processing tasks
- [Speech Translation](#speech-translation)

# Audio tasks
- [Speech To Text](#speech-to-text)

---

## Image Classification

- [Inception trained on ImageNet](./inception_imagenet/): Classify entire images into 1000 classes, like "Zebra", "Panda", and "Dishwasher".

## Object Detection

- [SSD MobileNet trained on Coco](./ssd_mobilenet_v2_coco/): Locate and classify objects into 80 classes with high speed.
- [FasterRCNN Resnet 50 trained Coco](./faster_rcnn_resnet50_coco/): Locate and classify objects into 80 classes with a high accuracy.
- [YOLOv2 trained Coco](./yolov2_coco/): Locate and classify objects into 80 classes with high speed.

## Pose Estimation

- [OpenPose trained on COCO Keypoint](./openpose_coco/): Find body keypoints (knees, arms, eyes, hip, etc.) in an image.

## Instance Segmentation

- [MaskRCNN Inception trained on COCO](./mask_rcnn_inception_v2_coco/): Segment all objects intances given a color image.

## Image Enhancement

- [Colorful Image Colorization](./zhang_colorization/): Colorize black and white images!

## Image Captioning

- [im2txt](./im2txt//): Caption an image. Implementation of [Show And Tell](https://arxiv.org/abs/1411.4555)


## Other Computer Vision  Models

- [Fast Style Transfer](./fast-style-transfer/): Add styles from famous paintings to any photo.

---

## Speech To Text

- [Deep Speech](./deep-speech/): Convert speech audio to text. Based on [mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech)

