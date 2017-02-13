**Vehicle Detection Project**

This is my (second) submission writeup for the vehicle detection project. This was actually an interesting project, though I wonder quite a bit about the details of improving the performance and running time of any solution, to be able to work on limited resources (as in an actual car). I suspect a FPGA implementation could do the trick (In fact, I see no other way to do this in a reasonable time!). Nevertheless, as a simple comment of mine, at some moment I tried to make a DNN version of this whole project, but that would have taken way way too long, though I did find a way (this is not rally difficult) in which it could reasonably work. But in the end, since that would have taken too long, I ended up using a simple SVM.

[//]: # (Image References)
[image1]: ./images/hog.png
[image2]: ./images/windows.png
[image3]: ./images/heatmapbase.png
[image4]: ./images/heatmap.png
[image5]: ./images/watershed.png
[image6]: ./images/boxes.png
[video1]: ./project_video.mp4

###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.
####2. Explain how you settled on your final choice of HOG parameters.

As usual, all my code is always available inside the IPYTHON notebook in this repo. I the case of hog, look at cell 6. The use of hog features is quite simple, I just applied the hog function from skimage.feature. I think the real information is about the chosen color space and parameter. Since, as some papers suggest (In particular, the one on the German traffic signs dataset by LeCun), the Y channel of the YUV space is usually a good pick. This has the advantage of allowing easier disctinction of the yellow color strips from the road. I did not even bother trying other color spaces because of that.

As per parameters, since the images were 64x64 pixels, 8x8 seemed a reasonable choice for pixels per cell, since it is the square root of the images dimensions. Also, I used 9 orientations, though, in retrospective, 8 could have worked well too (for pure symmetry). 

I also used a blocks per cell parameter of 2x2, but for this I have actually an interesting reason. Since at first I wanted to classify images into cars and not cars using a DNN, and some papers suggest that some of the very first layers of a heavily trained DNN seem to correspond with the orientations that are outputed by hog, I figured that I could just use the output from hog as some of the first layers of a DNN. But then I had a problem, since I did not know the precise meaning of the output from hog. So, I looked at the documentation (which was insufficient) and then I read the code and then I realized that, after reshaping, hog outputs the layers according to 'pans' of the cells in each block. So for example, if there are 2x2 cell per block, it is like applying a 'block operation' while moving the 'block' cell by cell over the original image, where the block just provides a metric for normalization of the results of the cells. So, since there are 4 cells in a block, each cell will be an output 4 times, because each cell can participate in 4 different blocks! An application of MaxPool seemed then obvious. But then, I had to devise a way to reshape the output from hog so this MaxPool application could be done easily. Turns out that, since the output from hog could not be reshaped as easily for what I wanted (because I would need to rearrange items in memory) and if the paramenter is cells per block parameter is 2x2, then it is really easy to simply extract the submatrices directly for each 'possibility' of block 'belonging' for each cell, and then I could very easily create an extended, plain 2d matrix (in effect a single layer) that holds ALL these arrangements, in such a way that a 2x2 MaxPool can select the 'best' cell among its repetitions, because all repetitions of a cell all lie within contiguous 2x2 squares! Neat, and very easy for 2x2, not so much for other choices (still possibly, but not as simple). But in the end I did not use any of that, so it is all converted to a vector.

This is an example output of a hog application:

![alt text][image1]

Of course I applies a mmild prepocessing, like per layer normalization, type checking (why is the type different for png?), I enabled the normalization (transform_sqrt) in hog as well. And of courde the hog functio wrapper automatically resizes input to a 64x64 image and applies the YUV color transform to get the Y channel.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the obtained hog features as obtained above, and also the color features of the image. In the case of the color features, this is simply the transformation of the image to other color spaces annexed as input. The color spaces I used were HSV (all channels) and RGB (also all channels). 

I also used histogram features. diving the color features obtained before into 32 bins. 32 seemed a good number, but there is little justification for this. 

Then I trained the SVM using all the images in the provided datasets, after calculating the features for all of them. I plotted the results of the SVM to see which threshold for the calculated distance by the SVM would be a good choice, to better prune false positives, and 0.4 seemed ok at first, but I had to raise it to 0.9 to deal with false positives when lowering the threshold for the heatmap, as in the first submission the white car kept on not being recognized at times.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Since I did not want to manually set some sort of search area and assign window sizes, I did this algorithmically. The idea is very simple, after idenifying the region where the centers of the cars could be (idealizing this as a trapezoid), I divide this region into horizontal segments. The separations of the segments define lines where to slide, horizontally, the search windows. The search windows size is then calculated linearly, assuimg certain size at the bottom line, an 0 at the top. For each size, 'variations' are allowed as well, as it could happen due to geometry or angle of the road that these estimates are wrong. Finally, a minimal size is set, so no search window can be less than this, to disallow unneeded applications of the search function. There is some degree of overlap allowed for the seach windows when sliding horizontally as well. 

I also tried a 'quadratic' (rather following a traingular series) algorithm for scaling the size of the search windows, but it did not work as well. However for the second submission I used both. I suspect a clever choice of windows could help a lot, and also, perhaps the heat map itself could hint locations for dense cluster of windows aiding in recognition.

So, for each search window, the entire pipeline described above is applied, the SVM distance is then thresholded, and an evaluation of 'carness' is provided.

![alt text][image2]

As for optimization, not much was done. I would need heavy parameter tuning for that, and I do not have time for that.

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I simply applied a heat map of ALL detected windows over the last 10 images, thresholded it and identified blobs within it using blob_log. To make it managable, I downscaled the heatmap image to 1/4 per side. This works quite well, for example, this is the heat map:

![alt text][image3]

And these are the detected blobs in the downscaled image.

![alt text][image4]

Then the heatmap is binarized with finer thresholds, and the centers of the blobs are used as starting point for watershed 'blob filling'.

![alt text][image5]

These 'blobs' are then used, together with the centers, to determine the bounding boxes assigned to each car. 

![alt text][image6]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Why is the white car so much harder to recognize? I suspect that the classifier is not as happy to pronounce it a car, since it may be a conflict with the white marker, or even cars wheels, or the sky. Again I hope a DCNN could do better, as it could 'easily' encode geometrical information of the cars. As you can see, even though, for the second submission, the car around the white car is generated even when afar, it still dissapears at some moments! I think I could increase the number of frames used even more, though that will increase the false positives too...

I think a DNN could do the car classification much more robustly, much it would take a long time to train, and so in the end I used and SVM. The approach is itself very simple, all the hog, color and color histogram features (for different color spaces) are transformed to signature vectors for each image. These are then normalized, and then feed to the SVM for training. The SVM is used on a sliding window approach, and the found car windows are then added to a heat map of the last 10 frames. The heatmap is thresholded, and the car blob are identified in it. The 'extension' of these blobs is then used to calculate bounding boxes for the cars, which are then added to the generated video.

This pipeline is fairly slow though. I do not see it failing very badly assuming constante illumination conditions, but it sometimes, very rarely, will still detect false positives. For this I suspect a DNN could work better, as false positives would be less probable (I believe). But processing time would then be even greater than my current implementation.

I do not do much for chosing the best features for the SVM, for example using a decision tree over the features could reduce their number, vastly speeding up my solution. It is also very obvious that fine parameter tuning is needed for thresholds, the sizes of the search windows, and so on.

Further on, I would try to use the output from a DNN instead of a SVM. I would also use the bounding boxes to better 'prune' the cars from the scene, and improve the previous project.