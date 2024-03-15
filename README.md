# UCSD HDSI - DSC 190 Winter 2024, Team #2

## Deliverables:

### Must have:
1. Model working on a pre-recorded video ✅

Original Video:

[![DSC 190 - Original Video](https://img.youtube.com/vi/74YdnvhRzvM/0.jpg)](https://www.youtube.com/watch?v=74YdnvhRzvM)

Processed Video 1 (Using 640x400p pre-processing, -15% and +15% brightness augmentations in training data):

[![DSC 190 - Processed Video 1](https://img.youtube.com/vi/74YdnvhRzvM/0.jpg)](https://www.youtube.com/watch?v=74YdnvhRzvM)

Processed Video 2 (Using -15% and +15% brightness augmentations in training data):

[![DSC 190 - Processed Video 2](https://img.youtube.com/vi/74YdnvhRzvM/0.jpg)](https://www.youtube.com/watch?v=74YdnvhRzvM)

Do note that we scaled our video down from 1920x1080p to 640x400p for faster processing on our CPU ([Intel Core i5-8250U](https://ark.intel.com/content/www/us/en/ark/products/124967/intel-core-i5-8250u-processor-6m-cache-up-to-3-40-ghz.html)). This also replicates the frames that our camera, the [Luxonis Oak-D LR](https://shop.luxonis.com/products/oak-d-lr), will process during the race.  

We originally pre-processed our training data to 640x400p because we feared that the model would not be able to predict on frames of a different size. The more pixelated training images, however, caused us to make inaccurate predictions as seen from the occasional holes that would appear in "Processed Video 1".

In "Processed Video 2", we trained a new model with training data that kept its original source resolution, and this gave us results that we deemed was acceptable. 

### Nice to have:
1. Model working on a Jetson AGX ❌
2. Model working on an Oak-D camera ❌

We really wanted to get the model running on the Jetson, but ran into issues with Roboflow recognizing the API key. 

Upon making a [post](https://discuss.roboflow.com/t/cant-run-semantic-segmentation-model-on-nvidia-jetson/4791) asking for help in the Roboflow forum, we received a response from a Roboflow employee suggesting we used their [Hosted Inference API](https://docs.roboflow.com/deploy/hosted-api) or their open-source [Inference package](https://inference.roboflow.com/) instead. This will be our next course of action when we continue working next quarter.

According to this [page](https://docs.roboflow.com/deploy/sdks/luxonis-oak), the Luxonis Oak does not yet support our model so we did not focus on getting the model working on the camera.

## Failed Attempts:

Most of our failed attempts were from trying to use past repos that did lane detection. Most of these models used OpenCV techniques which were interesting to read about, but took a lot of effort to tweak as most of the code available was outdated, and/or had a lot of parameters that we did not know how to adjust.

Something that we overlooked was that our model needed to differentiate between road and grass instead of lane and road; using the lane alone to detect edges was simply not reliable. 

We immediately knew that we had to look into supervised learning models such as YOLOv9 since we had such a unique scenario in which the road we are detecting is neighbored by grass.

## Working Attempts:

## Data Science Involvement:

### Data Collected:

We collected data using pictures taken by the Oak-D LR of the figure-8 in front of the UCSD Engineering building. We later abandoned these pictures because we received much higher quality [pictures](https://drive.google.com/drive/folders/16-9_a-NBHoKpIzlv7viFqjEUIDw6kFAW) from Jack of the Purdue track. 

### Data Science Techniques Used:

Although we did not choose to use OpenCV for our road detection model, we learned a lot about its application in computer vision. Some of these techniques are listed below:
1. [Hough Transform](https://github.com/cloudxlab/opencv-intro/blob/master/hough_transforms_lane_detection.ipynb)

Pros: 
- Less complex to implement
- Best at detecting straight lines
- Robust to noise and varying lighting conditions
Does not use machine learning

Cons:
- Can only detect straight lines. Have to use other techniques (Probabilistic Hough Transform, Hough Transform for circles) to detect curves

2. [HSV filtering and sliding window search 1](https://github.com/kemfic/Curved-Lane-Lines), 
[HSV filtering and sliding window search 2](https://github.com/galenballew/SDC-Lane-and-Vehicle-Detection-Tracking)

Pros:
- Incorporates color information
- Does not use machine learning
- Adaptable to changing lane positions

Cons:
- More complex to implement
- Sensitive to lighting conditions (e.g. direct sunlight)
- Requires careful parameter tuning for HSV color space
- Struggle with poor lane visibility

We also learned a great deal about computer vision in the context of deep learning through Roboflow. Semantic segmentation models were really new to us and we did not know that we could annotate images and image inferences. Our knowledge on training, validation and test sets in previous classes facilitated our understanding of the Roboflow framework.

### Data Analysis:

The only data analysis we did was assessing the quality of the data that we trained. Roboflow provided us an annotation heatmap which showed us how much we annotated our images. 
![alt text](image.png)

## Lessons Learned

## Future Plans