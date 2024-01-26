Face Detection in DeepfakeStream

Document stands for the overview of face detection approach, 
used in the DeepfakeStream project.

# Assumptions 
Project's goal is to finding deepfakes on human faces, 
processing raw video frames without specifying exact location of the faces
can generally lead to high computational costs and bad performance of the classifier.
Spotting faces on the big frame pictures and cropping them for further processing
is one the most efficient ways how to handle issue, without causing any additional 
overhead.

# Scope
We assume having a N number of video frames of quadratic size (NxN) with presence
of at least one human face of minimum size (MxM), where M should be set manually as 
it regulates the tradeoff between precision and computational costs. This is because
having smaller minimum face size introduces more potential examples for the network
to consider, therefore the overall computational time increase.

# Face detection network

We proposed usage of Multi-task Cascaded Convolutional Neural Network, also called "MTCNN", which is quite simple yet efficient model, that can be leveraged as a face detector.

<p align="center">
  <a><img src="https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/imgs/mtcnn/mtcnn_arch.jpeg" style="width: 50%; height: 50%"></a>
</p>

The overall architecture consists of 3 parts: Input Network (P-Net), Refine Network (R-Net) and Output Network (O-Net)

# Disadvantages

### 1. Inference time dependencies, which cannot estimate exactly

Model inference time cannot be inherently estimated, due to dependency not 
only on the size of the input image, but also on the number of faces, depicted on it.
You should clearly understand how many faces you can potentially encounter on a single image and try to fine tune the MTCNN, according to those assumptions, overwise it may lead to unexpected computation consumption increase.

### 2. Unholistic GPU parallelization support.

Model does not inherently full GPU parallelization. Each face, detected on the image is going to get through the same process of (Input, Refining, Output) which cannot be executed in parallel. Therefore, if you have 100 faces on the image, each of them will pass through the network's internals synchronously without opportunity of being processed in parallel, which contributes to overall increase in time.

