# Classifying Urban Land Use with a Convolutional Neural Network
## Simon Rimmele - sar2160
## E4040.2017Fall

This repository contains the codebase for a reproduction/analysis of the below paper:

>>  Using convolutional networks and satellite imagery to identify patterns in urban environments at a large scale. A. Toni Albert, J. Kaur, M.C. Gonzalez, 2017. In Proceedings of the ACM SigKDD 2017 Conference, Halifax, Nova Scotia, Canada.      

The github repository associated with the original paper can be found [here](https://github.com/adrianalbert/urban-environments).

### Organization

## Model Implementation and Results
All content is located in the Jupyter notebook.

## Code
There are several folders corresponding to the attribution of code components:

* **srimmele/** : This contains code chiefly written by me.
* **ecbm4040/**:  This contains code chiefly provided as course material, but is in some cases heavily modified by me.
* **UrbanCNN/**: This contains code stemming from the paper authors' git repository. It is primarily focused on collecting and preprocessing images and labels. In some cases it was modified for use in my environment.

Because of the attribution-based file structure, the code is not necessarily organized well in a conceptual or practical sense. I made the decision in order to  communicate as well as possible the authorship and origin of all code.

I have provided a "requirements.txt" since there are some image manipulation packages necessary to run the notebook. 

## Data and Imagery

**train.csv** and **test.csv** in the data/ folder contain class labels as well as the associated image filename.

Images are stored in imagery/, organized by city and land use label. The image data is too large to reside in the repository, but can be accessed in .zip format from a Google Storage Bucket using the code provided.

The ImageNet VGG16 weights used for transfer learning are also too large to store in the repository but the original file can be downloaded [here](http://www.cs.toronto.edu/~frossard/post/vgg16/).
