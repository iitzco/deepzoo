# Deep ZOO  

> How to find a model
> 1. Pick a **category**.
> 2. Pick the **task**
> 3. Navigate the options sorted by framework

# Categories:

This index will take you to all models of the category, regardless framework

- [Computer Vision](#computer-vision-tasks)
- [Natural Language Processing](#natural-language-processing-tasks)
- [Other](#other)

# Computer Vision tasks:

- [Image Classification](#image-classification)
- [Object Detection](#object-detection)
- [Pose Estimation](#pose-estimation)
- [Face Detection](#face-detection)
- [Instance Segmentation](#instance-segmentation)
- [Image Enhancement](#image-enhancement)
- [Other](#other-computer-vision-models)

# Natural Language Processing tasks:
- [Speech Translation](#speech-translation)

## Image Classification

- #### Tensorflow: 
    - [Inception trained on ImageNet](./inception_imagenet/): Classify entire images into 1000 classes, like "Zebra", "Panda", and "Dishwasher".

## Object Detection

- #### Tensorflow: 
    - [SSD MobileNet trained on Coco](./ssd_mobilenet_v2_coco/): Locate and classify objects into 80 classes with high speed.
    - [FasterRCNN Resnet 50 trained Coco](./faster_rcnn_resnet50_coco/): Locate and classify objects into 80 classes with a high accuracy.
    - [YOLOv2 trained Coco](./yolov2_coco/): Locate and classify objects into 80 classes with high speed.

## Pose Estimation

- #### Caffe: 
    - [OpenPose trained on COCO Keypoint](./openpose_coco/): Find body keypoints (knees, arms, eyes, hip, etc.) in an image.

## Instance Segmentation

- #### Tensorflow: 
    - [MaskRCNN Inception trained on COCO](./mask_rcnn_inception_v2_coco/): Segment all objects intances given a color image.

## Image Enhancement

- #### Caffe: 
    - [Colorful Image Colorization](./zhang_colorization/): Colorize black and white images!


## Other Computer Vision  Models

- #### Tensorflow: 
    - [Fast Style Transfer](./fast-style-transfer/): Add styles from famous paintings to any photo.

