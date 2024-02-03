# Fine tuning MTCNN Face Detector

Overview of fine-tuning process of MTCNN face detector

# Scope
The quality of the baseline MTCNN network decently sucks.
After initial check of performance on training dataset, it
generated significant number of FP (False Positive) crops,
which isn't tenable for our case, as those masks are then
used for core deepfake analysis, hence the overall success of the
project critically relies on performance of face detector. We want

# Resource requirements 

The overall forward pass for fine-tuning MTCNN can be calculated as: 
```
time = (MTCNN flops / GPU flops)
```

| module                 | #parameters or shape   | #flops    |
|:-----------------------|:-----------------------|:----------|
| MTCNN                  | 44.549M                | 88.837G   |

Required number of GPU flops can be approximated as
```
Total GPU FLOPS/s = clock_count * cores * flops_per_clock_cycle * fp_precision (default 32-bit, if you pick 64, then you are going to multiply by 0.5)
```
Hence, by having 88.837 GFlops per 32-bit input for MTCNN and expecting forward pass to take 0.1s, we need to have 888.37 GLops for a GPU.

In our case, we've selected 2 instances of GPU NVIDIA T4.

# Dataset capacity
We've labeled around ~1200 images of size (512x512x3) with selected precision of 32-bit.

# Dataset split
We've broken up dataset into 2 different splits for train and validation sets
using balanced sampling method. For training we've extracted around ~800 images, 250 for validation and 150 for early stopping regularization.

# Augmentations
We've used the same augmentations strategies, presented under 'docs/data_management/augmentations.md' document for train and validation dataset splits respectively.

# Distributed fine-tuning
We've leveraged `data parallelism` strategy for accelerating fine-tuning process of the model. The entire batch size of `64 samples` was equally splitted across 2 GPU instances. Then tuning were performed independently. Calculation of gradients is handled by `AllReduce` technique.

# Fine-tuning configuration
List of settings of best configuration, formed during experimentation
| parameter                 | parameter type   | value    |
|:-----------------------|:-----------------------|:----------|
| learning_rate                  | float               | 1e-5   |
| weight_decay                  | float               | 0.02   |
| batch_size                 | int               |  64  |
| max_epochs                  | int               | 20   |
| optimizer                 | -               | ADAM (Adaptive Moment Estimation)   |
| lr_scheduler              | -              | MultiStep Learning Rate Scheduler   |
| tuning device                  | -               | cuda, 2 x NVIDIA T4   |
| cpu threads for data loading                  | int               | 3   |
| loss function                 |  -               | CIOU Loss (Complete Intersection Over Union Loss)   |
| evaluation metric                  | -              | Average Precision (AP)  |
| distributed training                  | bool               | true   |


# Final results
The following results were achieved with the above setup
| metric                 | parameter type   | value    |
|:-----------------------|:-----------------------|:----------|
| CIOU Loss                 | float               | 0.27   |
| Average Precision (AP)                  | float               | 0.81   |

