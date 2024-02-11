# Deepfake Stream

DeepfakeStream is a web service for recognizing and spotting deepfakes on images and videos, artificially applied by Generative and Adversarial networks.

<p align="center">
  <a><img src="https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/imgs/preview/preview_deepfakestream.webp" style="width: 90%; height: 70%"></a>
</p>

## Table of Contents
-  Scope of the document
- Introduction
- Problem background
- Proposed solution
- Implementation plan 
- Testing and Validation 
- Deployment and Maintainance
- Considerations and Caveats 
- Conclusion
- Paper references

## Problem
Deepfake techniques been rapidly evolving nowadays. With the arise of GAN-based networks, the number of video face swapping methods have dramatically increased. This contributes to a subset of common problems, such as blackmailing people, generating fake or misleading news and many more.

To fight back against this emerging threat, we want to propose Deepfake classification pipeline, which is capable of identifying high-quality deepfakes, where swapping of human faces is tightly involved.

## Introduction
The goal of the project, is to correctly identify face swappings, also known as "deepfakes", presented on the video and generally minimize the number of false positives / negatives.

The input data typically passed as a video file (.mp4) with presence of (deepfaked / or not) human faces. The video then is splitted into independent segments, which serves as an input to face detector to extract human faces. After face extraction, it is passed through the CNN-based network, which identifies noise features by leveraging the concept of SRM filters, the output probability for the video is the average of probs across all segments.

## Background
We want to find solution, which adapts well to the `Deepfake Detection Challenge [DFDC] video dataset`, while demanding tenable amount of FLOPs and power to be inferenced in production environment realities.

## Proposed Solution

- **Architecture overview**: 

    The system comprises of a single REST application, deployed using containerization strategy called `Docker`, additionally, supports image builds via `CI` pipeline and deployment via corresponding `CD` pipeline, based on `Github Actions` technology.

- **Data ingestion and processing**:

    Before passing data right to the classifier, we crop out human faces to reduce number of unnecessary spatial features, then apply set of predefined augmentations. For more details, you can visit document, which 

- **Model Inference**:

    inference supports several options, which can be defined manually using configuration file. Currently, you can select from running network on `CPU` and one or multiple `GPUs`, other backends are not supported yet.

- **Model Design**: 
   
    Model consists of following stages:

    - MTCNN 
    - SRM Filter Convolution
    - EfficientNet-based Classifier
    - Custom CNN-based network

    We first pass data through `MTCNN`, to extract ROI (Regions of Interests), in our case, ROIs are represented as 'human faces'. Then, we compute noise features for each extracted face using concept of `SRM Filters`. After noise extraction, these maps are passed to EfficientNet-based classifier, which performs the main portion of noise analysis. Lastly, we pass the output of classifier to Custom CNN-based network, which computes the final probability of the face being deepfaked.

- **Data Collection**:

    As a main source of data, we used around 200G of real and deepfake videos, gathered from `Deepfake Detection Challenge (DFDC) Dataset`, collected by Meta. It contains 100K of deepfaked video clips with 1-2 random people per each, and 19K of real human video clips.

- **Evaluation Metrics**: 

    As an evaluation metric, we selected `F1-Score`.
    It provides the way of balancing between precision and recall. For deepfake detection problem, both precisio and recall are important. High precision indicates, that model accurately identifies positive examples, while preserves minimum number of false positives. High Recall indicates a high portion of accurately identified positive examples out of the entire distribution of positively marked samples. Therefore, `F1-Score` is highly valuable, as it ensures, that both objectives are met.

## Technologies and Languages

- Python (3.11.6) - prog. language for project dvlpmnt
- Torch (2.2.0) - DL Framework for Computer Vision
- Shell (3.2.57) - for writing efficient executable scripts
- Docker (20.10.12) - deployment strategy system
- Kubernetes - (1.27.10) - container orchestration system

## Improvements
Currently, the system can only accept images as an input, however in further patches we will add support for video and realtime stream processsing.

## Deepfake Paper References

- [Enhancing Deepfake Video Detection Using Data Augmentation Techniques by Akshat Savla, Sagar Sunil Dholakia](https://assets.researchsquare.com/files/rs-1844392/v1_covered.pdf?c=1659727359)

- [Learning Rich Features for Image Manipulation Detection by Peng Zhou, Xintong Han](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2813.pdf)

- [Understanding the Deepfake Detectability of Various Affinity Groups Using A Dual Stream Network by Rohit V Gopalakrishnan](https://www.cs.columbia.edu/~jrk/NSFgrants/videoaffinity/Interim/22x_Rohit.pdf)

- [Image prefiltering in Deepfake Detection by Szymon Motłoch, Mateusz Szczygielski and Grzegorz Sarwas](https://www.scitepress.org/Papers/2022/108412/108412.pdf)

- [The Deepfake Detection Challenge (DFDC) Preview Dataset](https://arxiv.org/abs/1910.08854)

- [Improved Fakes and Evaluation of the State of the Art in Face Manipulation Detection](https://arxiv.org/pdf/1911.05351.pdf?)

## Other references
- ["Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression" by Zhaohui Zheng, Ping Wang](https://arxiv.org/abs/1911.08287)
- ["Rethinking model scaling of Convolutional Neural Networks" by Mingxing Tan, Quoc V. Le](https://arxiv.org/abs/1905.11946)
- ["A Survey of Convolutional Neural Networks: Analysis, Applications, and Prospects" by Zewen Li, Wenjie Yang, Shouheng Peng, Fan Liu](https://arxiv.org/abs/2004.02806)
- [Global Weighted Average Pooling Bridges Pixel-level Localization and Image-level Classification](https://arxiv.org/abs/1809.08264)
- ["Focal Loss: Better way for appoaching unbalanced classification problems" by Lin et al.](https://paperswithcode.com/method/focal-loss)

## Appendices
Here you can find additional documentation to gain more clearity on how project works under the hood.

- [Data Augmentaions](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/data_management/AUGMENTATIONS.md)
- [Model interpretability](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/interpretability/PRED_EXPLAINABILITY.md) 

- [Model testing and validation](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/model_validation)

- [Model training](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/training/MODEL_TRAINING.md)

- [Data preprocessing](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/data_management/DATA_MANAGEMENT.md)

- [Drift](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/maintainance/DRIFT.md)

- [Deployment](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/deployment/DEPLOYMENT.md)

- [Face detection](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/face_detection/DETECTOR.md)

- [Monitoring](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/maintainance/MONITORING.md)

- [API Design](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/system/API_DESIGN.md)

- [Infrastructure Design](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/system/INFRASTRUCTURE_DESIGN.md)

- [Continious Integration & Deployment](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/ci_cd)

- [Project timeline](https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/TIMELINE.md)
