import unittest
from model import OnnxModel

class TestOnnxModel(unittest.TestCase):
    def test_model_loading(self):
        model = OnnxModel("model.onnx")
        self.assertIsNotNone(model.ort_session)

    def test_predict(self):
        model = OnnxModel("model.onnx")
        output = model.predict("n01440764_tench.jpg")
        self.assertEqual(len(output), 1000)

if __name__ == "__main__":
    unittest.main()
