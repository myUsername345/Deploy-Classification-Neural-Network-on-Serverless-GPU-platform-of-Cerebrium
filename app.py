from flask import Flask, request, jsonify
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

class OnnxModel:
    def __init__(self, onnx_model_path):
        self.session = onnxruntime.InferenceSession(onnx_model_path)

    def predict(self, image_path):
        image = ImagePreProcessor.preprocess_image(image_path)
        inputs = {self.session.get_inputs()[0].name: image}
        result = self.session.run(None, inputs)
        return result[0]

class ImagePreProcessor:
    @staticmethod
    def preprocess_image(image_path):
        image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = preprocess(image).unsqueeze(0)
        return image.numpy()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image provided'}), 400

    image_path = "n01440764_tench.jpg"
    file.save(image_path)

    try:
        onnx_model_path = "path_to_your_model.onnx"
        model = OnnxModel(onnx_model_path)
        prediction = model.predict(image_path)
        predicted_class_id = np.argmax(prediction)
        return jsonify({'predicted_class_id': int(predicted_class_id), 'probabilities': prediction.tolist()}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
