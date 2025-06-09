import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class OnnxModel:
    def __init__(self, onnx_model_path):
        self.ort_session = ort.InferenceSession(onnx_model_path)

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = preprocess(image).unsqueeze(0).numpy()
        return image_tensor

    def predict(self, image_path):
        image_tensor = self.preprocess(image_path)
        ort_inputs = {self.ort_session.get_inputs()[0].name: image_tensor}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return ort_outs[0]

if __name__ == "__main__":
    model = OnnxModel("model.onnx")
    output = model.predict("n01440764_tench.jpg")
    print(output)
