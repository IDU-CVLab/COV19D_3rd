[![DOI:10.1007/978-3-031-62269-4_31](http://img.shields.io/badge/DOI-10.1080/21681163.2023.2219765-B31B1B.svg)](https://doi.org/10.1007/978-3-031-62269-4_31)  
# Database
* This study coincides with the third run of the [IEEE ICASSP 2023: AI-enabled Medical Image Analysis Workshop and Covid-19 Diagnosis Competition (AI-MIA-COV19D)](https://mlearn.lincoln.ac.uk/icassp-2023-ai-mia/). An extended version of COV19-CT-DB was used.
* IDU-CVLab came ***fourth*** on the [Leaderboard](https://drive.google.com/file/d/1ATt-sqsSSaQczz-Qxj85LohwPD3T0i3W/view)

# Method
* The main purpose of this study is to propose an accurate framework for COVID-19 detection from a big and challenging database of Computed Tomography (CT) images. The proposed pipeline includes a segmentation part, a region-of-interest extraction part, and a classification part. The code in this repo includes the whole pipeline.
* The segmentation part can be found [here](https://github.com/IDU-CVLab/Images_Preprocessing_2nd). 
* In the classification part, a Convolutional Neural Network (CNN) was used to take the final diagnosis decisions.
* Please see the attached paper for full details.  
* The robustness of the method was checked by applying the framework on noise-added images of the orignal ones and validating the results using the validation partition of the data. Salt-and-paper we well as Gussian noise types were used to check the robustness of our method.The code for that is named "Noisey-Images-Segmentation-Classification-Framework.py".
   
# Web-apps
Web-apps were deployed using streamlit (you may need to wait for the application to start. Please refresh the page if it takes long to start the application):  
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
If you find this work useful, kindly cite our chapter in the book proceedings:  

>@InProceedings{10.1007/978-3-031-62269-4_31,  
author="Morani, Kenan  and Ayana, Esra Kaya and Kollias, Dimitrios and Unay, Devrim",  
editor="Arai, Kohei",  
title="Detecting COVID-19 in Computed Tomography Images: A Novel Approach Utilizing Segmentation with UNet Architecture, Lung Extraction, and CNN Classifier",  
booktitle="Intelligent Computing",  
year="2024",  
publisher="Springer Nature Switzerland",  
address="Cham",  
pages="450--465",  
abstract="Our study introduces an innovative framework tailored for COVID-19 diagnosis utilizing a vast, meticulously annotated repository of CT scans (each comprising multiple slices). Our framework comprises three key Parts: the segmentation module (based on UNet and optionally incorporating slice removal techniques), the lung extraction module, and the final classification module. The distinctiveness of our approach lies in augmenting the original UNet model with batch normalization, thereby yielding lighter and more precise localization, essential for constructing a comprehensive COVID-19 diagnosis framework. To gauge the efficacy of our framework, we conducted a comparative analysis of other possible approaches. Our novel approach segmenting through UNet architecture, enhanced with Batch Norm, exhibited superior performance over conventional methods and alternative solutions, achieving High similarity coefficient on public data. Furthermore, at the slice level, our framework demonstrated remarkable validation accuracy and at the patient level, our approach outperformed other alternatives, surpassing baseline model. For the final diagnosis decisions, our framework employs a Convolutional Neural Network (CNN). Utilizing the COV19-CT Database, characterized by a vast array of CT scans with diverse slice types and meticulously marked for COVID-19 diagnosis, our framework exhibited enhancements over prior studies and surpassed numerous alternative methods on this dataset.",  
isbn="978-3-031-62269-4"  
}  

