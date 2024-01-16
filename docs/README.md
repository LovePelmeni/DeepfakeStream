
# DeepStream face detector

Welcome to the design document for our Online RESTful Machine Learning-powered service dedicated to detecting deepfake manipulations in videos, specifically focusing on human faces. As the prevalence of deepfake technology continues to rise, the need for robust and scalable solutions to identify manipulated content has become crucial. Our goal is to develop a state-of-the-art service that leverages advanced machine learning algorithms to detect and mitigate the impact of maliciously created deepfake videos.

# Documentation

I've listed up list some of the design documents, which you can read and analyze for deeper understanding of problem context, project API and system design, model and data nuances, other technical details, relevant to system's lifecycle.

- **API_DESIGN.md** - overview on the REST API of the service and how to interact with different routes.
- **MONITORING.md** - overview on system and model / data monitoring, embraces nuances on how model, data and system health is monitored, as well, as metrics being leveraged for achieving it.
- **PRED_EXPLAINABILITY.md** - overview on model interpretation, which techniques are used for ensuring, that model leverages relevant features for predicting labels.
- **DATA.md** - overview of the data, used for training the model. Includes feature analysis, image properties and augmentation strategies
- **MODELING.md** - overview on modeling analysis.
- **TECHNICAL_STACK.md** - overview on technical stack, used for the project.