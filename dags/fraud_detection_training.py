import os
import logging 
from dotenv import load_dotenv

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers = [
        logging.FileHandler('./fraud_detection_model.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# class that is used for the model training 
class FraudDetectionTraining:
    def __init__(self, config_path = '/app/config.yaml'):
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet',
        os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = '/usr/bin/git',

        load_dotenv(dotenv_path = '/app/.env')

        self.config = self.load_config(config_path)

        os.environ.update({
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'AWS_S3_ENDPOINT_URL': self.config['mlflow']['s3_endpoint_url']
        })
        
        # ensure our environment is ready before training
        self.validate_environment()

        mlflow.set_tracking_uri(self.config['mlflow']['tracking_url'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        
    # function to load in config.yaml
    def load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info('Configuration loaded successfully')
            return config
        
        except Exception as e:
            logger.error('Failed to load configuration: %s', str(e))
            raise 
    
    # function to validate the environment
    def validate_environment(self):
        

# 1. read data from Kafka/read in the data 

# observe the label of the data: what % of the data is fradulent and not fraudulent 

# address the class imbalance by using boosting methods so the model can recognize both classes

# conduct hypterparameter tuning by outputting a confusion matrix to see how the model is performing 

# use the metrics from the model to compare the performance of the current model with the past models to use the best potential model 

