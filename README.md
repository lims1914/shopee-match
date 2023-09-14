# shopee-match

For our project we will use Shopee’s Price Match Guarantee Dataset from Kaggle competition. The dataset
contains more than 30000 images and corresponding text description of these images. For the text file, 
we have five columns, the first and second column represent the id code and image name. The
third column is a perceptual hash of the image. The fourth column is the description of the image, and the
last column is the label. One of the most important features about this dataset is the difference between
related products may be subtle while the image of these products may be wildly different. The data can be
download from https://www.kaggle.com/c/shopee-product-matching/. 



This project is written in Pytorch.

dataset.py: contains class about how to process image dataset, text dataset and clip model dataset./
model.py: the image model and text model. The image model contains effecientnet and nfnet./
get_embedding.py: extract image embedding and text embedding usimg models in model.py. extract embedding using clip model/
main.py: main function to get results/
utils: contains some useful functions


The result is shown below






![image](https://github.com/lims1914/shopee-match/assets/40879123/715c8da5-26fb-4014-9c59-b1647b19df7d)

