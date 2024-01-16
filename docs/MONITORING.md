# Monitoring DeepfakeStream Service

The purpose of this document is to outline the monitoring strategy of this Machine Learning Service. The primary objective is to ensure, that the model, data or server performance do not degrade over time, even as data becomes larger, evolves in quality or distribution.

# Scope 
We want to encompass the lifecycle of the model, data it was trained on, and the service itself,
which provides connectivity with users via HTTP. As data evolves, the model
may not be able to handle it properly, due to the drastical change in size, content drift,
or other type of common shifts. Additionally, we want to track, whether we have increase in traffic or server load, to have an opportunity to respond in time without ruining user experience.

# Assumptions

We assume, that the service runs on the platform, which supports integration with Prometheus or Graphana. We use it's API urls for storing and parsing metrics for further analysis. Down below, we've provided a small guidiance on how to access and interact with this functionality offline.


## Online metrics (model-specific)

1. **Model inference time in (ms)** - for the given batch of data 
(can be reset manually via .env configuration)
2. **Number of returned NaNs** - number of times model returned None

## Implemenation of the online model-specific metrics.


## Online metrics (data-specific)

1. Label distribution drift metric. (check of the label distribution and how it changes over time)
2. Feature drift metric. (check presence of important features and how it's number evolve over time)
3. Prediction drift metric. (evaluating image on previously gathered validation set)

## Online Metrics (hardware metrics)

1. CPU utilization (percentage).
2. GPU utilization (percentage).
3. RAM utilization (in megabytes).
4. Network bandwith.

## Online metrics (server baseline-specific)

1. Average HTTP request session time.
2. Unique visitors.




