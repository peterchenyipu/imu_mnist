import torch
from models.resnet1d import IMUMinstResNet1D, SmallResNet1D, resnet18_1d, SimpleClassifier, SmallResNet1D1Layer
from models.append_softmax import AppendSoftMax
import onnx
import onnxruntime

# Load the trained model
# model_dict = torch.load('best_model.pth')
# model = IMUMinstResNet1D(num_classes=10, in_channels=6)
# model.load_state_dict(model_dict)

# model = SmallResNet1DWithReshape(num_classes=10, dropout_rate=0.0)
# model = AppendSoftMax(model)
# model = SimpleClassifier(num_classes=10)
model = IMUMinstResNet1D(num_classes=10, dropout_rate=0.5)
model.load_state_dict(torch.load('best_model.pth'))
model = AppendSoftMax(model)
model.eval()

# Export the model to ONNX, fix the batch size to 1
dummy_input = torch.ones(1, 1800)
print(f'model inference result: {model(dummy_input)}')

torch.onnx.export(model, dummy_input,
                  'model.onnx',
                  opset_version=11,
                  input_names=['input'],
                  output_names=['output'])

# load the model and run inference
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession('model.onnx')
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)

print(f'onnx inference result: {ort_outs[0]}')
