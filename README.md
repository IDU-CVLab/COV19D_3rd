[![DOI:10.48550/arXiv.2210.02992](http://img.shields.io/badge/DOI-10.1101/2021.01.08.425840-B31B1B.svg)](https://doi.org/10.48550/arXiv.2210.02992)

# Introduction
* The main purpose of this study is to propose a pipeline for COVID-19 detection from a big and challenging database of Computed Tomography (CT) images. The proposed pipeline includes a segmentation part, a region of interest extraction part, and a classifier part. The code in this repo includes the whole pipeline.
* The segmentation part can be found [here](https://github.com/IDU-CVLab/Images_Preprocessing_2nd). 
* In the classification part, a Convolutional Neural Network (CNN) was used to take the final diagnosis decisions.

# Dependencies
Numpy == 1.19.5 </br>
CV2 == 4.5.4 </br>
Tensorflow == 2.5.0 </br>

# Cite
If you find the Code helpful, please consider citing the preprint at:
@article{morani2022covid,
  title={COVID-19 Detection Using Segmentation, Region Extraction and Classification Pipeline},
  author={Morani, Kenan},
  journal={arXiv preprint arXiv:2210.02992},
  year={2022}
}
