#Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/sliding.png
[image2]: ./output_images/sliding1.png
[video1]: ./output.mp4



###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook Vehicle-Detection.ipynb`). 

HOG features has been explored using skimage function like `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`)

Finally we choosed the following values of HOG features:
* color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
* orient = 5  # HOG orientations
* pix_per_cell = 8 # HOG pixels per cell
* cell_per_block = 5 # HOG cells per block
* channel = "ALL" # Can be 0, 1, 2, or "ALL"
* spatial_size = (16, 16) # Spatial binning dimensions
* hist_bins = 24    # Number of histogram bins
* spatial_feat = False # Spatial features on or off
* hist_feat = False # Histogram features on or off
* hog_feat = True # HOG features on or off

####2. Explain how you settled on your final choice of HOG parameters.

HOG parameters based on classifier accuracies and looking at the output images. Though we can use HOG parameters as a hyperparameters and use GridCV search too. But, this is done fairly manully.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features.

Following pipeline has been used to train the model.
* Format features using `np.vstack` and `StandardScaler()`.
* Split data into shuffled training and test sets
* Train linear SVM using `sklearn.svm.LinearSVC()`.

I trained a linear SVM using all the HOG extracted features of all the channels and using YcrBcr. Which imporoved the accuracy to 98%.

###Sliding Window Search

Sliding window is just a windowing function or a convolution in whole image to search for the car and if it is found we make a box. Searching fairly starighforward based on the heat map of boxes, we selected that area.


Initially I trained the model using singla channel HOG Features , the performance of the model was around 95 %, then by including all the features. it increased. Instead of RGB , using YcrBr, proved to be better as accuracy went from 97 % to 98%.

![alt text][image1]
![alt text][image2]
---

### Video Implementation

The whole pipeline that has been developed for a single image, is used to the whole video where each frame of the video is being used.
Here's a [link to my video result](./project_video.mp4)


---

###Discussion

* There were instances where our classifier didn't perform good as it struggled to identify the car, By using more data to train would help.
* We can take the cache the past frames and use them too for better identification.




After few experiments, with channels of HOG, using all channels proved to be better as compared to use either of one. We could also , go for principal component analysis , which would help us choose features better and also, speed our training process and thus the whole pipeline.