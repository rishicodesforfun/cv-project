from pathlib import Path
import json
import unittest
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]


class TestPlantDiseaseProject(unittest.TestCase):

    def test_model_exists(self):
        model_path = ROOT / "models" / "plant_disease_model.keras"
        self.assertTrue(model_path.exists())

    def test_class_names_exist(self):
        class_file = ROOT / "models" / "class_names.json"
        self.assertTrue(class_file.exists())

    def test_class_count(self):
        class_file = ROOT / "models" / "class_names.json"

        with open(class_file, "r") as f:
            classes = json.load(f)

        self.assertEqual(len(classes), 10)

    def test_model_prediction_shape(self):
        model_path = ROOT / "models" / "plant_disease_model.keras"
        model = tf.keras.models.load_model(model_path)

        test_image = np.zeros((1, 160, 160, 3), dtype=np.float32)

        prediction = model.predict(test_image, verbose=0)

        self.assertEqual(prediction.shape, (1, 10))

        self.assertAlmostEqual(
            float(np.sum(prediction)),
            1.0,
            places=3
        )


if __name__ == "__main__":
    unittest.main()