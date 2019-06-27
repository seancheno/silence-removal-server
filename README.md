# Silence Removal 

Silence removal is a RESTful-API powered by Flask that removes non speech/music from mp3 and wav files using a custom trained Convolutional Neural Net. The model was built and trained with keras using 10,000 silence and non-silence wav samples extracted from the TensorFlow Speech Recognition Challenge [public dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data) using a custom audio processing script.

* This is the server-side repo powered by Flask.

* The client-side repo built with React can be found [here](https://github.com/seancheno/silence-removal-client).

* Visit the live demo at [silenceremoval.com](http://silenceremoval.com).


## Installation

	# Clone the repo
    git clone https://github.com/seancheno/silence-removal-server/
    cd silence-removal-server
    
    # Create and activate Python 3.7 virtual environment
    python3.7 -m virtualenv venv
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Start the server
    python app.py   
   

## Notes

The server stores the user-uploaded audio files in AWS S3, so before running `python app.py`, add your S3 bucket configuration information in the `config.py` file.

## Model Summary
An existing model architecture found [here](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py) was used and tweaked by adding additonal conv2d layers. 
_________________________________________________________________
Layer (type)                 Output Shape              Param #
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 20, 1, 32)         64
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 1, 48)         1584
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 20, 1, 120)        5880
_________________________________________________________________
depthwise_conv2d_1 (Depthwis (None, 20, 1, 120)        240
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 20, 1, 155)        18755
_________________________________________________________________
dropout_1 (Dropout)          (None, 20, 1, 155)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 20, 1, 155)        0
_________________________________________________________________
dropout_2 (Dropout)          (None, 20, 1, 155)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 3100)              0
_________________________________________________________________
dense_1 (Dense)              (None, 128)               396928
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256
_________________________________________________________________
dropout_4 (Dropout)          (None, 64)                0
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 130
_________________________________________________________________


* **Total params: 431,837** 
* **Trainable params: 431,837**
* **Non-trainable params: 0**

_________________________________________________________________