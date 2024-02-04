# DeepfakeStream

Design documentation for the machine learning-powered service responsible for detecting deepfake manipulations in videos, specifically focusing on human faces. As the prevalence of deepfake technology continues to rise, the need for robust and scalable solutions to identify manipulated content has become crucial. Our goal is to develop a state-of-the-art service that leverages advanced machine learning algorithms to detect and mitigate the impact of maliciously created deepfake videos.

<p align="center">
  <a><img src="https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/imgs/srm/srm_features.png" style="width: 70%; height: 70%">SRM based approach for deepfake detection</a>
</p>

## Deepfake detection pipeline

For extended pipeline details, checkout series 
of design documents under `docs/deepfake_detection` folder.

<p align="center">
  <a><img src="https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/imgs/architecture/pipeline_architecture.png" style="width: 70%; height: 70%">Architecture of the pipeline.</a>
</par

---
# Documentation

I've listed up list some of the design documents, which you can read and analyze for deeper understanding of problem context, project API and system design, model and data nuances, other technical details, relevant to system's lifecycle.

- **API_DESIGN.md** - overview on the REST API design of the service and how to interact with application via HTTP protocol.
- **MONITORING.md** - overview on system and model / data monitoring, embraces nuances of how model, data and system health is monitored, as well, as metrics being leveraged for achieving it.
- **PRED_EXPLAINABILITY.md** - overview on model interpretation, which techniques are used for ensuring, that model leverages relevant features for predicting labels.
- **DATA.md** - overview of the data, used for training the model. Includes feature analysis, image properties and augmentation strategies.
- **MODELING.md** - overview on modeling analysis.
- **TECHNICAL_STACK.md** - overview on technical stack, used for the project.
- **DEPLOYMENT.md** - overview on project's deployment process, key files and strategies.
- **DRIFT.md** - overview on statistical / non-statistical methods, used for detecting data drift inside the system.
- **AUGMENTATIONS.md** - overview on image augmentations and transformations, applied to the data.
- **FACE_DETECTION.md** - overview of the method, used for detecting human faces on images.


# Reference links of papers, used during development

- https://www.cs.columbia.edu/~jrk/NSFgrants/videoaffinity/Interim/22x_Rohit.pdf
- https://iopscience.iop.org/article/10.1088/1757-899X/533/1/012054/pdf
- https://arxiv.org/abs/1604.02878
- https://hal.science/hal-04206611/document
- https://arxiv.org/abs/2302.12445
- https://arxiv.org/abs/1909.02061
