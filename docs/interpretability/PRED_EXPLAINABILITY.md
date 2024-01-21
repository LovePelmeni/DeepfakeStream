# Network prediction interpretation.

Distinguishing between real human faces and deepfakes 
has never been an easy task to address. That's why it's crucial 
to understand the strategy your model leverages for predicting final labels,
to ensure it's reliability. Down here, I've specified some of the popular
techniques, which worked decently well for highlighting influential areas
on images.

# Visual Interpretation Methods

### Gradient-based Class Application Mapping (GradCAM)

One of the common and powerful methods for interpreting 
deep learning networks, which been empirically proven 
to work effetively with not only visual data, but also with audio.

Original Paper for more details: https://arxiv.org/abs/1610.02391

Real example of image interpretation:

<p align="center">
  <a><img src="https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/imgs/grad_cam/grad_cam_analysis.png" style="width: 50%; height: 50%"></a>
</p>

Here, the area of importance is indicated by the color of the gradient.
Blue represents the least used features, while red stands for the most
important and highly impactful ones, which drives the network 
to make the prediction the most. 

# Other considerations 

Other ubiqutious methods, like KernelSHAP (Kernel Shapley Values)
and LIME (Local Interpretable Model Agnostic Explanations) ended up
not working well enough, and giving less interpretable explanations
in contrast with GradCAM.

# Implementation
You can find the project's source code with integration under:
    - `src/explainers/explainers.py`
