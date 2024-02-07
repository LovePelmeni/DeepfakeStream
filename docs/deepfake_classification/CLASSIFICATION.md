# Deepfake Classification

# Assumptions

# Approach

We created custom classifier, based on the approach,
proposed in paper "". Lastly we've added 3D SRM Filter, 
which captures the noise of the input image, and then
passes noisy map to the main classifier.

<p align="center">
  <a><img src="https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/imgs/classification/CNN_CLASSIFIER.png" style="width: 90%; height: 70%"></a>
</p>

## Inference Parallelization 

RGB and SRM Stream Networks can be parallelized
during the inference phase and synchronize the results
at the fusion stage, which brings faster , yet preserving the same accuracy.

### Considerations and Caveats


# References 
Paper, which explains internals of Dual Stream Net: https://www.cs.columbia.edu/~jrk/NSFgrants/videoaffinity/Interim/22x_Rohit.pdf