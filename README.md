# Language Classifier

Binary Language Classifier with PyTorch and Flask.

Dataset: https://www.kaggle.com/datasets/basilb2s/language-detection

<br />

## Local Setup

### Install requirements
    pip install -r requirements.txt

### Debug
    python src/app.py

### Run
    SET FLASK_APP=./src/app.py 
    python -m flask run

<br />

## Docker Setup

### Build Docker Image
    docker build . -t language-classifier-image

### Run Docker Container
    docker run --name language-classifier -d -p 5000:5000 language-classifier-image

<br />

## **Usage**

### **Train** [POST]
Endpoint: `http://127.0.0.1:5000/train`

All hyperparameters are optional and, if not set, default values are used. Below is an example of a **body request** with the configurable hyperparameters and their default values:

**Request**
```json 
    {
        "epochs": 10,
        "lr": 5, 
        "step_size": 1.0,
        "gamma": 0.1,
        "batch_size": 64,
        "input_dim": 4,
        "embed_dim": 32,
        "num_classes": 17,
        "eval_every": 100
    }
```
**Response**
```json 
    {
        "accuracy": 0.90
    }
```

<br />

### **Test** [GET]
Endpoint: `http://127.0.0.1:5000/test`

**Response**
```json 
    {
        "test_accuracy": 0.90
    }
```

### **Inference** [POST]
Endpoint: `http://127.0.0.1:5000/predict`

**Request**
```json 
    {
        "text": "questa è una frase in italiano!"
    }
```
**Response**
```json 
    {
        "class": 1
    }
```

<br />

### **TensorBoard** [GET]

    docker cp language-classifier:/app/runs docker_runs
    tensorboard --logdir docker_runs

<br />

### **Preprocessing Automated Tests** [GET]

    pytest src/tests.py

<br />

### **Binary to Multi-Class**
The model is configured to recognize Italian sentence. To switch to Multi-Class configuration change this line in `predict()` method:
```python
response = 1 if LANG_LOOKUP[label.item()] == "Italian" else 0
```
to
```python
response = LANG_LOOKUP[label.item()]
```