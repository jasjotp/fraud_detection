import os 
import logging
import pickle 
from pyspark.sql import SparkSession
from dotenv import load_dotenv
import yaml

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)

# class for the inference to use our model to predict on new data in real time coming in from Kafka
class FraudDetectionInference:
    bootstrap_servers = None 
    topic = None
    security_protocol = None
    sasl_mechanism = None 
    username = None
    password = None 
    sasl_jaa_config = None 

    # create the initialization function 
    def __init__(self, config_path = '/app/config.yaml'):
        load_dotenv(dotenv_path = '/app/.env')
        self.config = self.load_config(config_path)
        self.spark = self.init_spark_session()
        self.model = self.load_model(self.config['model']['path'])
        self.broadcast_model = self.spark.sparkContext.broadcast(self.model)
        logger.debug(f'Environment variables loaded: {dict(os.environ)}')

    # function that loads the model 
    def load_model(self, model_path):
        try:
            # try to open the model path 
            with open(model_path, 'r') as f:
                model = pickle.load(f)
                logger.info('Model loaded from: {model_path}')
                return model
        except Exception as e:
            logger.error('Error loading model: {e}')

    # function to load config for Kafka 
    @staticmethod
    def load_config(config_path):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f'Error loading the config file: {e}')
            raise 
    
    # function to initialize the Spark Session 
    def init_spark_session(self):
        try:
            packages = self.config.get('spark', {}).get('packages', '')
            if packages:
                builder = SparkSession.builder.appName(self.config.get('spark').get('app_name', 'FraudDetectionInference'))
                builder = builder.config('spark.jars.packages', packages)
            spark = builder.getOrCreate()
            logger.info('Spark session initialized')
            return spark

        except Exception as e:
            logger.error(f'Error initiailizing Spark Session: {e}')
            raise

if __name__ == "__main__":
    inference = FraudDetectionInference(config_path = '/app/config.yaml')
    inference.run_inference()
