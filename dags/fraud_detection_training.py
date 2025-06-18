import os
import logging 
import boto3
import yaml
import mlflow
import pandas as pd
import json
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
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
        os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = '/usr/bin/git'

        load_dotenv(dotenv_path = '/app/.env')

        self.config = self.load_config(config_path)

        os.environ.update({
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'AWS_S3_ENDPOINT_URL': self.config['mlflow']['s3_endpoint_url']
        })
        
        # ensure our environment is ready before training
        self.validate_environment()

        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
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
        required_vars = ['KAFKA_BOOTSTRAP_SERVERS', 'KAFKA_USERNAME', 'KAFKA_PASSWORD']

        # check for missing variables, if we are not able to get any of these parameters, raise an error
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f'Missing required environment variables: {missing}')
        
        # if the variable is not missing, cheeck the minio connection 
        self.check_minio_connection()
    
    # function to check the minio connection 
    def check_minio_connection(self):
        try:
            s3 = boto3.client(
                's3',
                endpoint_url = self.config['mlflow']['s3_endpoint_url'],
                aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            )

            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            logger.info('Minio connection verified. Buckets: %s', bucket_names)

            # get the mlflow bucket, if there is none, default to a bucket called mlflow
            mlflow_bucket = self.config['mlflow'].get('bucket', 'mlflow')

            # if the bucket does not exist, create the bucket in S3W
            if mlflow_bucket not in bucket_names:
                s3.create_bucket(Bucket = mlflow_bucket)
                logger.info('Created missing MLFlow bucket: %s', mlflow_bucket)
        except Exception as e:
            logger.error('Minio connection failed: %s', str(e))
    
    # function to read in data from Kafka
    def read_from_kafka(self) -> pd.DataFrame:
        try:
            topic = self.config['kafka']['topic']
            logger.info(f'Connecting to Kafka Topic: {topic}')

            # ingest data from Kafka using the consumer
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers = self.config['kafka']['bootstrap_servers'].split(','),
                security_protocol = 'SASL_SSL',
                sasl_mechanism = 'PLAIN',
                sasl_plain_username = self.config['kafka']['username'],
                sasl_plain_password = self.config['kafka']['password'],
                value_deserializer = lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset = 'earliest',
                consumer_timeout_ms = self.config['kafka'].get('timeout', 10000)
            )

            # read in the message from the consumer 
            messages = [msg.value for msg in consumer]
            consumer.close()

            df = pd.DataFrame(messages)

            if df.empty:
                raise ValueError('No message received from Kafka.')

            df['timestamp'] = pd.to_datetime(df['timestamp'], utc = True)
            
            # is the is_fraud label is missing, we are missing the label of the data, so raise an error
            if 'is_fraud' not in df.columns:
                raise ValueError('Fraud label (is_fraud) missing from Kafka data')
            # if a transction is classed as non-fraud but is actually fraudulent, we reclassify that transaction as a fraudulent transsaction, so we retrain the model on the new classification
            
            # find the fraud rate and log it (percentage of fraudulent transactions)
            fraud_rate = df['is_fraud'].mean() * 100
            logger.info(f'Kafka data read successfully with fraud rate: {fraud_rate:.2f}')

            return df 

        except Exception as e:
            logger.error(f'Failed to read data from Kafka: {str(e)}', exc_info = True)
            raise 
    
    # function to create features that the model can train on 
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        # sort the user id and timestamp in order in ascending order, so we know who performed each transaction first
        df = df.sort_values(['user_id', 'timestamp']).copy()

        # extract temporal features 
        # hour of transaction 
        df['transaction_hour'] = df['timestamp'].dt.hour

        # flag transactions happening at night (between 10PM and 5AM)
        df['is_night'] = ((df['transaction_hour'] >= 22) | df['transaction_hour'] < 5).astype(int)

        # flag transactions happening on the weekend (Saturday = 5, Sunday = 6)
        

    # create a function to train the model
    def train_model(self):
        try:
            logger.info('Starting model training process...')

            # read in transaction data from Kafka
            df = self.read_from_kafka()

            # feature engineering of our raw data into categorical and numerical variables that our model will use 
            data = self.create_features(df)

        except Exception as e:
            pass 

        # split the data into train and test data 

        # log model and artifacts in MLFlow

        # preprocessing pipeline 

        # observe the label of the data: what % of the data is fradulent and not fraudulent 

        # address the class imbalance by using boosting methods so the model can recognize both classes

        # conduct hypterparameter tuning by outputting a confusion matrix to see how the model is performing 

        # use the metrics from the model to compare the performance of the current model with the past models to use the best potential model 

