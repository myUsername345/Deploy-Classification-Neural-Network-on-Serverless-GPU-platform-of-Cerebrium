import torch
import torch.onnx
from model import ClassificationModel

def convert_to_onnx('pytorch_model_weights.pth', 'model.onnx'):
    model = ClassificationModel()
    model.load_state_dict(torch.load('model.onnx'))
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, 'model.onnx', input_names=['input'], output_names=['output'], opset_version=11)
    print(f"Model has been converted to ONNX and saved at {onnx_path}")