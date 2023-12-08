
# DeepStream face detector

Welcome to the design document for our Online RESTful Machine Learning-powered service dedicated to detecting deepfake manipulations in videos, specifically focusing on human faces. As the prevalence of deepfake technology continues to rise, the need for robust and scalable solutions to identify manipulated content has become crucial. Our goal is to develop a state-of-the-art service that leverages advanced machine learning algorithms to detect and mitigate the impact of maliciously created deepfake videos.

# Technologies

**Main Programming Language:** Python (v3.10)

**Modeling:** Pytorch (v)

**Image Augmentation libraries:** albumentations (v), torchvision (v)

**Analytics and Data Management:** opencv2 (v), numpy (v), pandas (v)

**Plot and Visualization:** matplotlib (v), seaborn (v)

**Deployment:** FastAPI (v)

# Data

Common input to the application is base64 string, representing image. However for training, we've used 20G of raw videos, with different people depicted on them, including variation in rase, gender and age, to make dataset more diverse for training.


### Collection

### Preprocessing

Preprocessing include several steps:

1. Augmentation
2. Cutout Regularization
3. Isotropic Resizing

### Note
If you are interested in scrutinizing any of the aforementioned stages, recommend to look at `this paper`, which provides view on selected augmentations, as well, as regularization methods for the network

# Modeling

### Loss function

### Evaluation metrics

# Monitoring

## Assestment metrics

for evaluating such a ML system, we've splitted metrics into several layers.

### Online metrics (server monitoring)
After countless number of read papers, the final set looks like this: 
1. `Server Load` (%) (avg number of requests per second at a given time)
2. `Request Time` - (time in ms for HTTP-Request to be fully completed)

### Input Metrics (model input metrics)
Set of input specific metrics, which tracks the change 
in image distribution over time. 

1. Average image brightness in (%).

2. `Image distribution` (histogram) properties: 
    `mean`, `standard deviation`, `variance`

### Output Metrics (model output metrics)

1. `Number of times`, model returned None.
2. `Latency` of the network `inference` on a given set of images.

## Logging

For logging actions, happening inside the application, 
we use standard TXT files, which are stored inside the application, through it's lifecycle.

## Healthcheck
for regular healthecking of the application, we created 
separated HTTP (GET) Endpoint, which saves set of health metrics, everytime it is being called

## Drift Detection 

### Data Drift

### Concept Drift


## Public API Reference

#### Get all items

```http
  GET /predict/human/deepfake/
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `image` | `base64-encoded string` | **Required**. Base64 Encoded image with deepfake|

## Internal API Reference


```http
  GET /healthcheck/
```

| Description|
| :-------- |
| API Endpoint for parsing current health state of the application. **Requires CORS Authentication** |


```http
  GET /system/resource/consumption/
```

| Description                |
| :------------------------- |
| API Endpoint for parsing information about service usage. **Requires CORS Authentication**|
# Deployment


# Feedback