# X-ray Body Part Classifier

### Contents:
- ``x-ray-image-classification.ipynb``: exploring training and testing the model
- ``submission.py``: classify the test dataset and save the results to sample submission using the trained model
- ``model.tflite``: trained tensorflow lite modell, contains labels as metadata
- ``model_edgetpu.tflite``: model compiled for the Coral edge TPU

### Usage: 
``x-ray-image-classification.ipynb`` contains training the model with labeling and displaying a set of images from the test dataset.
``submission.py`` uses the trained modell to label all images in the test dataset and saves it in the data submission file to submit to kaggle, it also works with the Coral edge TPU if the ``use_tpu`` variable is set to ``True``

[original dataset in DICOM format](https://www.kaggle.com/competitions/unifesp-x-ray-body-part-classifier/data),
[png converted dataset used for training](https://www.kaggle.com/datasets/ibombonato/xray-body-images-in-png-unifesp-competion)