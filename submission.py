import os
import zipfile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

modelpath = 'model.tflite'
use_tpu = False

# get labels from model metadata
zip = zipfile.ZipFile(modelpath)
label_file = zip.read('labels.txt')
labels = label_file.decode("utf-8").splitlines()

# import the right packages dipending on the use of cpu or tpu
from tflite_runtime.interpreter import Interpreter
if use_tpu:
    from tflite_runtime.interpreter import load_delegate

if use_tpu:
    import platform
    EDGETPU_SHARED_LIB = {
        'Linux': 'libedgetpu.so.1',
        'Darwin': 'libedgetpu.1.dylib',
        'Windows': 'edgetpu.dll'
    }[platform.system()]
    interpreter = Interpreter(model_path='model_edgetpu.tflite',
                                experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB)])
else:
    interpreter = Interpreter(modelpath)

#allocate the tensors
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_index = input_details[0]["index"]

def predict(image):
    resized_img = cv2.resize(image,(input_shape[1], input_shape[2]))
    input_tensor= np.array(np.expand_dims(resized_img,0))

    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)

    return (pred.argmax(), pred.max()/255)

def predict_samples(row):
    image = cv2.imread(f'images/test/{row["SOPInstanceUID"]}-c.png')
    predict_class, accuraccy = predict(image)

    row['Target'] = predict_class

tqdm.pandas()
sample_submission_df = pd.read_csv('sample_submission.csv')
sample_submission_df.progress_apply(predict_samples, axis=1)
sample_submission_df.to_csv('sample_submission.csv', index=False)
