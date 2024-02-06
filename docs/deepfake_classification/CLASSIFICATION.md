# Deepfake Classification


# Assumptions

# Approach

## Dual Stream Network

One of the yet common solutions for face manipulation analysis, which paralellizes the processes of visual and spatial artifacts detections is called `Dual (Two) Stream Network`.

<p align="center">
  <a><img src="https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/imgs/classification/dual_stream_net.webp" style="width: 100%; height: 100%"></a>
</p>

It separately runs two independent CNN-based networks, one, 


## Inference Parallelization 

RGB and SRM Stream Networks can be parallelized
during the inference phase and synchronize the results
at the fusion stage, which brings faster , yet preserving the same accuracy.

### Considerations and Caveats


# References 
Paper, which explains internals of Dual Stream Net: https://www.cs.columbia.edu/~jrk/NSFgrants/videoaffinity/Interim/22x_Rohit.pdf