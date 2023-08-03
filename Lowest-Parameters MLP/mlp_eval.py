import numpy as np
import tflite_runtime.interpreter as tflite
import pandas as pd

# =================================================

model_file = "LPmlp.tflite"			        # model path
test_data_file = "../test_data_odd.csv"	    # test data path

# =================================================

# set model interpreter and allocate tensors
interpreter = tflite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
# print(input_shape) # check input_shape of the model : (1, 128)

# load test data
data = pd.read_csv(test_data_file)
test_X = data.drop('128', axis=1)
test_Y = data['128']

test_X, test_Y = test_X.to_numpy(), test_Y.to_numpy()
test_X = test_X.astype(np.float32)

# resize input tensor to infer all test data at once
interpreter.resize_tensor_input(input_details[0]['index'], test_X.shape)
interpreter.allocate_tensors()

interpreter.set_tensor(input_details[0]['index'], test_X) # set test X as input tensor

# predict
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index']) # get output tensor
output_data = np.argmax(output_data, axis =-1)  # get predicted label from output tensor
corr_cnt = (output_data==test_Y).sum()          # get number of correct answers
acc = float(corr_cnt)/float(test_X.shape[0])      # calculate accuracy

print("test accuracy:", acc)