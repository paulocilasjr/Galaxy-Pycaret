import pickle
import sys
import logging

from pycaret_classification import ClassificationModelTrainer
from pycaret_regression import RegressionModelTrainer

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

input_file = sys.argv[1]
target_col = sys.argv[2]
output_dir = sys.argv[3]
model_type = sys.argv[4]
model_file = sys.argv[5]

# Load your pre-trained model
with open(model_file, 'rb') as file:
    pretrained_model = pickle.load(file)

if model_type == "classification":
    trainer = ClassificationModelTrainer(input_file, target_col, output_dir)
    trainer.evaluate(model_file)
elif model_type == "regression":
    trainer = RegressionModelTrainer(input_file, target_col, output_dir)
    trainer.evaluate(model_file)
