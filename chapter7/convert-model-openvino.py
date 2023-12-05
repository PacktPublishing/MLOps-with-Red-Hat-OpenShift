# this python file will take the tensor flow model and convert it into the penvino format

import openvino as ov
import tensorflow as tf

core = ov.Core()

# optionally convert h5 model into tensorflow model
import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
tf.saved_model.save(model,'model')

# reading from the model that was saved during the training notebook
ov_model = ov.convert_model("./model")
ov.save_model(ov_model, 'model.xml')
