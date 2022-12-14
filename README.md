[![DOI:10.48550/arXiv.2210.02992](http://img.shields.io/badge/DOI-10.1101/2021.01.08.425840-B31B1B.svg)](https://doi.org/10.48550/arXiv.2210.02992)

# Introduction
* This study coincide wityh the third run of the [IEEE ICASSP 2023: AI-enabled Medical Image Analysis Workshop and Covid-19 Diagnosis Competition (AI-MIA-COV19D)](https://mlearn.lincoln.ac.uk/icassp-2023-ai-mia/)
* The main purpose of this study is to propose a pipeline for COVID-19 detection from a big and challenging database of Computed Tomography (CT) images. The proposed pipeline includes a segmentation part, a region of interest extraction part, and a classifier part. The code in this repo includes the whole pipeline.
* The segmentation part can be found [here](https://github.com/IDU-CVLab/Images_Preprocessing_2nd). 
* In the classification part, a Convolutional Neural Network (CNN) was used to take the final diagnosis decisions.

# Web-apps
* Web-apps were deployed: <br/> 
&nbsp; - from a single 2D grayscale medical slice [here](https://kenanmorani-covid-19deployment-pipeline-app-82q4v6.streamlit.app/)   
&nbsp; - from a full CT scan image [here](https://kenanmorani-covid-19deployment-patient-level-predictions-d37izn.streamlit.app/)

# Dependencies
numpy==1.21.6 </br>
tensorflow==2.9.2 </br>
keras==2.9.0 </br>
streamlit==1.14.0 </br>
scipy==1.7.3 </br>
scikit_image </br>
opencv-python-headless </br>


# Cite
If you find the Code helpful, please consider citing the preprint at: </br>
@article{morani2022covid,   </br>
  title={COVID-19 Detection Using Segmentation, Region Extraction and Classification Pipeline},     </br>
  author={Morani, Kenan},    </br>
  journal={arXiv preprint arXiv:2210.02992},      </br>
  year={2022}     </br>
}
