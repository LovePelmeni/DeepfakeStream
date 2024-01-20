# REST Application Interface Design

This document provides overview on rest api design of 
the project, covers available endpoints, how to properly
interact with them and which data expect as an input.

# Idea

We want to expose our network to the real world, so it can interact
with other users via suitable interface. 

# Scope

This document does not cover deploying nuances / strategies
or any other aspects of the system design. The main 
idea is to provide guide on how to access model, when it's ready to be deployed.
# Assumptions

We assume model is not running on edge and have access
to high-bandwith network. 

## Public API Reference 


Analyze single video frame (image)

```http
  GET /api/scan/image/frame/
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `image_base64_hash`  | `string` | **Required**. Base64 encoded image. |

Outputs HTTP response with status code of execution and json-encoded context.
Which contains following information:

- deepfake_prob - (fp16) - probability of image being deepfaked.
- human_prob - (fp16) - probability of image being an actual human face photo.


## Internal API Reference (for monitoring)

```http
  GET /api/monitoring/health/
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `id`      | `string` | **Required**. Id of item to fetch |

#### add(num1, num2)

Takes two numbers and returns the sum.


```http
  GET /api/monitoring/server/load
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `id`      | `string` | **Required**. Id of item to fetch |

#### add(num1, num2)

Takes two numbers and returns the sum.


```http
  GET /api/monitoring/resource/consumption
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `id`      | `string` | **Required**. Id of item to fetch |

#### add(num1, num2)

Takes two numbers and returns the sum.


