# shopee-match

For our project we will use Shopee’s Price Match Guarantee Dataset from Kaggle competition. The dataset
contains more than 30000 images and corresponding text description of these images. The shape of the image
is not fixed, ranging from 640*640 to 1024*1024, so we need to unify them during training and testing. For
the text file, we have five columns, the first and second column represent the id code and image name. The
third column is a perceptual hash of the image. The fourth column is the description of the image, and the
last column is the label. One of the most important features about this dataset is the difference between
related products may be subtle while the image of these products may be wildly different. The data can be
download from https://www.kaggle.com/c/shopee-product-matching/
