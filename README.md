


# ImageProcessing
- Untitled11.py : This script is for inference detection using custom trained Fast RCNN, EfficientDet, and SSDMobilenet models.

- untitled13.py : This script is for inference detection using custom trained Mask RCNN models (i.e. the segmentation models).

- untitles12.py : This script is for inference detection using models such as Fast RCNN, EfficientDet, Mask RCNN and SSDMobilenet models which are pretrained on the Microsoft COCO datatset.

- YoloV3.ipynb : This python notebook is for running inference using a YoloV3 model which makes use of COCO pre-trained weights.  

- untitled17.py : This script is for detecting the edges of the water surface using canny edge detection algorithm and then it determines the depth of the water using the aspect ratio concept. 

**If you find our Environmental Modelling & Software paper or code useful, we encourage you to cite the paper. BibTeX:**

`@article{PALLY2021105285,
title = {Application of image processing and convolutional neural networks for flood image classification and semantic segmentation},
journal = {Environmental Modelling & Software},
pages = {105285},
year = {2021},
issn = {1364-8152},
doi = {https://doi.org/10.1016/j.envsoft.2021.105285},
url = {https://www.sciencedirect.com/science/article/pii/S1364815221003273},author = {R.J. Pally and S. Samadi}`

# Project Title
FloodImageClassifier: A Python tool for Flood Image Classification and Semantic Segmentation

## Getting Started
FloodImageClassifie.py is a python package developed using python 3.9. The package is tested using >9000 image data collected from the USGS, 511 traffic images (DOT) and social media platforms. The user can specifically target any flood image and use any of these pre-trained models to estimate the inundation area and flood depth severity. 

### Prerequisites 

The package dependencies are:            
*  os
*  numpy
*  pathlib
*  tensorflow
*  time
*  PIL
*  cv2
*  matplotlib
*  warnings
*  math
*  skimage.viewer
*  imutils

## Authors


* **Rakshit Pally** 
* **Dr. Vidya Samadi** 


## Acknowledgments

* This work is supported by the U.S. National Science Foundation (NSF) Directorate for Engineering under grant # CBET 1901646. Any opinions, findings, and discussions expressed in this study are those of the authors and do not necessarily reflect the views of the NSF. 

