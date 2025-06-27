import os 
import logging
import pickle 
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType,
                              IntegerType, DoubleType, TimestampType)
from pyspark.sql.functions import (from_json, col, hour, dayofmonth,
                                  dayofweek, when, lit, coalesce, window)
from dotenv import load_dotenv
import yaml
import json

from dags.fraud_detection_training.py import read_from_kafka

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
    
    # helper function to read data from Kafka
    def read_from_kafka(self):
        # load Kafka configuration parameters with fallback values
        logger.info(f'Reading data from Kafka topic: {self.config['kafka']['topic']}')
        kafka_bootstrap_servers = self.config['kafka']['bootstrap_servers']
        kafka_topic = self.config['kafka']['topic']
        kafka_security_protocol = self.config['kafka'].get('security_protocol', 'SASL_SSL')
        kafka_sasl_mechanism = self.config['kafka'].get('sasl_mechanism', 'PLAIN')
        kafka_username = self.config['kafka'].get('username')
        kafka_password = self.config['kafka'].get('password')

        # configure Kafka SASL authentication string
        kafka_sasl_jaas_config = (
            f'org.apache.kafka.common.security.plain.PlainLoginModule required '
            f'username="{kafka_username}" password="{kafka_password}";'
        )

        # store configuration details in instannce variables for reuse 
        self.bootstrap_servers = kafka_bootstrap_servers
        self.topic = kafka_topic
        self.security_protocol = kafka_security_protocol
        self.sasl_mechanism = kafka_sasl_mechanism
        self.username = kafka_username
        self.password = kafka_password
        self.sasl_jaas_config = kafka_sasl_jaas_config
        
        # use readStream to read from Kafka
        df = (self.spark.readStream
              .format('kafka') 
              .option('kafka.bootstrap.servers', kafka_bootstrap_servers) 
              .option('subscribe', kafka_topic) 
              .option('startingOffsets', 'latest') 
              .option('kafka.security.protocol', kafka_security_protocol) 
              .option('kafka.sasl.mechanism', kafka_sasl_mechanism)  
              .option('kafka.sasl.jaas.config', kafka_sasl_jaas_config)
              .load()
            )

        # define the JSON schema for the incoming transactions 
        json_schema = StructType([
            StructField('transaction_id', StringType(), nullable = True),
            StructField('user_id', IntegerType(), nullable = True),
            StructField('amount', DoubleType(), nullable = True),
            StructField('currency', StringType(), nullable = True),
            StructField('merchant', StringType(), nullable = True),
            StructField('timestamp', TimestampType(), nullable = True),
            StructField('location', StringType(), nullable = True)
        ])

        # parse the transactions that come in
        parsed_df = (df.selectExpr('CAST(value AS STRING)')
                    .select(from_json(col('value'), json_schema).alias('data'))
                    .select('data.*')
                    )

        # return the parsed and streamed data 
        return parsed_df 
    
    def get_user_activity_24h(self, df):
        '''
        Finds the number of transactions made by each user in the past 24 hours using a sliding window.
        Applies a 25-hour watermark on the event time column to allow Spark to handle late-arriving data
        and safely manage state. Groups events by user_id and a 24-hour window that slides every minute,
        then counts the number of events per user in each window and renames the count column to 'user_activity_24h'
        '''
        # set the data that is to be streamed to be able to be up to 25 hours late, so Spark waits 25 hours (an extra hour after the 24 hour rolling window) for late data, so no data older than the max event time - 25 hours arrives anymore
        df_with_watermark = df.withWatermark('timestamp', '25 hours')

        # group by the 24-hour sliding window + user_id
        user_activity_24h = (
            df_with_watermark
            .groupBy(
                window(col('timestamp'), '24 hours', '1 minute'),
                col('user_id'))
            .count()
            .withColumnRenamed('count', 'user_activity_24h')
        )

        # join back with the main data on user_id and timestamp falling into the same window (start date of window <= timestamp and end date of window > timestamp)
        combined_df = (
            df.join(
                user_activity_24h, 
                on = ((col('df.user_id') == col('user_activity_24h.user_id')) &
                     (col('df.timestamp') >= col('user_activity_24h.window.start')) &
                     (col('df.timestamp') < col('user_activity_24h.window.end'))),
                how = 'left'
            )
            .drop('user_activity_24h.user_id')
            .drop('window')
        )

        return combined_df

    # function to add features into the dataframe 
    def add_features(self, df):
        df = df.withColumn('transaction_hour', hour(col('timestamp'))) # hour the transaction occurred
        df = df.withColumn('is_weekend', 
                        ((dayofweek(col('timestamp')) == 1) | (dayofweek(col('timestamp')) == 7)).cast('int')) # flag whether the tranaction happend on a weekend or not 
        df = df.withColumn('is_night',
                           (hour(col('timestamp')) >= 22) | (hour(col('timestamp')) < 5)).cast('int') # whehter the transaction happen overnight (between 10PM and 5AM)
        df = df.withColumn('transaction_day', dayofweek(col('timestamp')))
        df = self.get_user_activity_24h(df) # counts the number of transacions for each user in a rolling 24 hour window using timestamp

    # function to run infernence on the model so it is trained on new data that comes in 
    def run_inference(self):
        df = self.read_from_kafka()
        df = self.add_features(df)

if __name__ == "__main__":
    inference = FraudDetectionInference(config_path = '/app/config.yaml')
    inference.run_inference()
