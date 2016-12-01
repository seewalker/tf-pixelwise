##A tensorflow implementation of Fully Convolutional Networks For Semantic Segmentation.

#USAGE

To run the trained classifier on some images:

python wrap_pixelwise.py run COCO 7 pix7 -imgdir=\<directory\>

This will iterate over images in \<directory\> and using a model trained on 72 known classes...
1) save visualizations of the predictions in the demo-results directory. These start with the prefix 'net-'.
2) save further visualizations in the demo-results direcotry showing a breakdown by which layers the predictions are coming from (recall that in the paper, the final predictions are a weighted average of predictions from a few layers: the earlier layers having higher resolution and the later layers having higher-level features). These start with the prefix 'comparison-'.
3) print out the top-k categories associated with an image along with the name of the image.

None is a category because this model has been trained to distinguish foreground from background.

#INSTALLATION

The only software dependencies are the various python modules being imported and CUDA. It is tested on CUDA 7.5 and with python 3.5.
In terms of hardware, a GPU will help, but tensorflow should be able to failover into CPU-only mode.
