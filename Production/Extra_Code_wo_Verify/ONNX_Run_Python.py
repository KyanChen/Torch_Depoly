import onnxruntime
import numpy as np

session = onnxruntime.InferenceSession("onnx_model.onnx")

input_name = session.get_inputs()[0].name
# print("input name", input_name)
output_name = session.get_outputs()[0].name
# print("output name", output_name)

numpy_array_img = np.random.randn(1, 3, 1024, 1024)
# forward model
res = session.run([output_name], {input_name: numpy_array_img})
# return a list
pred_seg = np.array(res)
pred_seg = np.argmax(pred_seg, axis=1) #(B, 1024,1024)