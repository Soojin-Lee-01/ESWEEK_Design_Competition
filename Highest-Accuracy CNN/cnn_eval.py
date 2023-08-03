import numpy as np
import tflite_runtime.interpreter as tflite
import pandas as pd

# =================================================

model_file = "HAcnn.tflite"			        # model path
test_data_file = "../test_data_even_odd.csv"	# test data path

# =================================================

# set model interpreter and allocate tensors
interpreter = tflite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
# print(input_shape) # check input_shape of the model : (1, 32, 4)

# load test data
data = pd.read_csv(test_data_file)
test_X = data.drop('128', axis=1)
test_Y = data['128']

test_X, test_Y = test_X.to_numpy(), test_Y.to_numpy()

test_shape = test_X.shape # shape of loaded test data : (length of data, 128)

test_X = np.array_split(test_X, 4, axis=1) # split 4 sensor data
test_ax, test_ay, test_az, test_str = test_X[0], test_X[1], test_X[2], test_X[3]

test_X = np.zeros((test_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
test_X[..., 0], test_X[..., 1], test_X[..., 2], test_X[..., 3] = test_ax, test_ay, test_az, test_str    # merge 4 sensor data
del test_ax, test_ay, test_az, test_str

# print(test_X.shape) # check shape of test data : (length of data, 32, 4)

# resize input tensor to infer all test data at once
interpreter.resize_tensor_input(input_details[0]['index'], test_X.shape)
interpreter.allocate_tensors()

interpreter.set_tensor(input_details[0]['index'], test_X) # set test X as input tensor

# predict
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index']) # get output tensor
output_data = np.argmax(output_data, axis =-1)  # get predicted label from output tensor
corr_cnt = (output_data==test_Y).sum()          # get number of correct answers
acc = float(corr_cnt)/float(test_shape[0])      # calculate accuracy

print("test accuracy:", acc)