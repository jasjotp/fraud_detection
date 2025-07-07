import os 
import logging
import joblib
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType,
                              IntegerType, DoubleType, TimestampType, LongType)
from pyspark.sql.functions import (from_json, col, hour, dayofmonth, pandas_udf, PandasUDFType, udf, dayofweek, when, lit, coalesce, window, avg, unix_timestamp, lag, count)
from pyspark.sql.window import Window
from redis import Redis
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
import yaml
import json
from redis_utils import (get_redis_connection, redis_activity_udf, get_amount_to_avg_ratio_redis)

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
    sasl_jaas_config = None 

    # create the initialization function 
    def __init__(self, config_path = '/app/config.yaml'):
        load_dotenv(dotenv_path = '/app/.env')
        self.config = self.load_config(config_path)
        self.spark = self.init_spark_session()
        self.model = self.load_model(self.config['model']['path'])
        self.broadcast_model = self.spark.sparkContext.broadcast(self.model)
        tracking_uri = self.config["mlflow"]["tracking_uri"]  
        mlflow.set_tracking_uri(tracking_uri)
        self.mlflow_client = MlflowClient(tracking_uri = tracking_uri)
        logger.debug(f'Environment variables loaded: {dict(os.environ)}')

    # function that loads the model 
    def load_model(self, model_path):
        try:
            # try to load the model path 
            model = joblib.load(model_path)
            self.selected_features = getattr(model, 'selected_features_', None)
            
            if self.selected_features is None:
                raise ValueError("Model file has no 'selected_features_' attribute - Retrain the model with the patch that saves it.")
    
            logger.info(f'Model loaded from: {model_path} with {len(self.selected_features)} input features.')
            return model
        except Exception as e:
            logger.error(f'Error loading model: {e}', exc_info = True)
            raise

    # function to load config for Kafka 
    @staticmethod
    def load_config(config_path):
        try:
            with open(config_path, 'rb') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f'Error loading the config file: {e}')
            raise 
    
    # function to initialize the Spark Session 
    def init_spark_session(self):
        try:
            builder = SparkSession.builder.appName(self.config.get('spark', {}).get('app_name', 'FraudDetectionInference'))

            packages = self.config.get('spark', {}).get('packages', '')
            if packages:
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
        logger.info(f"Reading data from Kafka topic: {self.config['kafka']['topic']}")
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
    
    # function to find rolling 24-hour user activity
    def get_user_activity_24h(self, df):
        '''
        Returns a streaming DF with columns
        window, user_id, user_activity_24h
        (24 h window sliding every minute).
        '''
        # set the raw event data that is to be streamed to be able to be up to 25 hours late, so Spark waits 25 hours (an extra hour after the 24 hour rolling window) for late data, so no data older than the max event time - 25 hours arrives anymore
        return (
            df.withWatermark('timestamp', '25 hours')
            .groupBy(
                window(col('timestamp'), '24 hours', '1 minute'),
                col('user_id')
            )
            .agg(count('*').alias('user_activity_24h'))
        )
    
    # function to send the user_id: user_activity_24h key value aggregation to Redis for storage
    def write_activity_to_redis(self, batch_df, _):
        redis_conn = get_redis_connection()

        for row in batch_df.select('user_id', 'user_activity_24h').distinct().collect():
            redis_conn.set(f"user:{row['user_id']}:activity24h", row['user_activity_24h'])

    # function to batch wrtite each user's last timestamp to redis so we can use a UDF to find the time since the last transaction without using a Window function in Spark
    def write_last_ts_to_redis(self, batch_df, _):
        r = get_redis_connection()
        for row in batch_df.select('user_id', 'timestamp').collect():
            r.set(f"user:{row['user_id']}:last_ts", 
                  str(row['timestamp'].timestamp())
            )
    
    # function to add features into the dataframe 
    def add_features(self, df):
        df = df.withColumn('transaction_hour', hour(col('timestamp'))) # hour the transaction occurred

        df = df.withColumn('is_weekend', 
                        ((dayofweek(col('timestamp')) == 1) | (dayofweek(col('timestamp')) == 7)).cast('int')) # flag whether the tranaction happend on a weekend or not 
        
        df = df.withColumn('is_night',
                           ((hour(col('timestamp')) >= 22) | (hour(col('timestamp')) < 5)).cast('int')) # whehter the transaction happen overnight (between 10PM and 5AM)
        
        df = df.withColumn('transaction_day', dayofmonth(col('timestamp')))

        df = df.withColumn('user_activity_24h', redis_activity_udf(col('user_id'))) # counts the number of transacions for each user in a rolling 24 hour window using timestamp
        df = df.withColumn('amount_to_avg_ratio', get_amount_to_avg_ratio_redis(col('user_id'), col('amount'))) # get the ratio of the current amount to the rolling mean of the last 6 transactions (exluding current one) 
        
        # create a feaure that flags whether the merchant is a high risk merchant or not 
        high_risk_merchants = self.config.get('high_risk_merchants', ['QuickCash', 'GlobalDigital', 'FastMoneyX'])
        df = df.withColumn('merchant_risk',
                           when(col('merchant').isin(high_risk_merchants), lit(1)).otherwise(lit(0)))

        return df 
    
    # helper function to get the most recent thereshold for the most recent run in MLFlow
    def get_threshold(self, experiment_name: str) -> float:
        client = self.mlflow_client
        experiment = client.get_experiment_by_name(experiment_name)
        
        # if there is no experiemnt name matching the parameter passed in, raise an error 
        if experiment is None:
            all_names = [exp.name for exp in client.search_experiments()]
            raise ValueError(
                f'Experiment: {experiment_name} not found on {client.tracking_uri}. '
                f'Available Experiments: {all_names}'
                )

        # retrieve the most recently completed run 
        runs = client.search_runs(
            experiment_ids = [experiment.experiment_id],
            filter_string = "attributes.status = 'FINISHED'",
            order_by = ["start_time DESC"], # sort so the newest run comes first and limit result to 1 so we get the most recent run
            max_results = 1
        )

        # if there are no completed runs found, raise an error 
        if not runs:
            raise ValueError('No completed runs found in experiment')
        
        latest_run = runs[0]

        # try to get a threshold from metrics first, then fallback to params
        if 'threshold' in latest_run.data.metrics:
            return float(latest_run.data.metrics['threshold'])
        
        full_run = client.get_run(latest_run.info.run_id)
        threshold = full_run.data.params.get('threshold')
        if threshold is None:
            raise ValueError('Threshold not found in the latest run')
        
        return float(threshold)

    # function to run infernence on the model so it is trained on new data that comes in 
    def run_inference(self):

        df = self.read_from_kafka()

        # launch side pipeline that maintains user_activity_24h and writes the updated actiivty per user to Kafka
        user_activity_24h_df = self.get_user_activity_24h(df)

        self.activity_query = (
            user_activity_24h_df.writeStream
            .outputMode('update')
            .foreachBatch(self.write_activity_to_redis)
            .queryName("write_user_activity_to_redis")
            .option("checkpointLocation", "checkpoints/activity_to_redis")
            .start()
        )

        # side pipleine that maintains the last timestamp per user and writes updated timestamps for each user to Kafka
        (
            df.select('user_id', 'timestamp')
            .writeStream
            .foreachBatch(self.write_last_ts_to_redis)
            .outputMode('update')
            .option("checkpointLocation", "checkpoints/last_ts_to_redis")
            .start()
        )
        
        logger.info(f"Successfully read data from Kafka topic: {self.config['kafka']['topic']}")
        df = self.add_features(df)
        logger.info("Successfully added features to dataframe")

        broadcast_model = self.broadcast_model
        threshold = self.get_threshold(experiment_name = 'fraud_detection_')

        # select the selected features from Optuna so we can use then inside the below UDF
        selected = self.selected_features 

        # create a udf function to predict the value (0/1) of the transaction
        @pandas_udf('int', PandasUDFType.SCALAR)
        def predict_udf(*cols: pd.Series) -> pd.Series:
            # list out the schema that we want to predict on, as a pandas df
            input_df = pd.concat(cols, axis = 1)
            input_df.columns = selected

            # get probabilities of the fraud cases 
            prob = broadcast_model.value.predict_proba(input_df)[:, 1]
            predictions = (prob >= threshold).astype(int)

            return pd.Series(predictions)
        
        # predict on the new transactions that come in
        predict_cols = [col(c) for c in self.selected_features]
        prediction_df = df.withColumn('prediction', predict_udf(*predict_cols))

        # only filter for fraud predictions that are flagged as fraudulent
        fraud_predictions = prediction_df.filter(col('prediction') == 1)
        fraud_predictions.selectExpr(
            "CAST(transaction_id as STRING) AS key",
            "to_json(struct(*)) as value"
        ).writeStream \
        .format('kafka') \
        .option('kafka.bootstrap.servers', self.bootstrap_servers) \
        .option('topic', 'fraud_predictions') \
        .option('kafka.security.protocol', self.security_protocol) \
        .option('kafka.sasl.mechanism', self.sasl_mechanism) \
        .option('kafka.sasl.jaas.config', self.sasl_jaas_config) \
        .option('checkpointLocation', 'checkpoints/fraud_predictions') \
        .outputMode('update') \
        .start() \
        .awaitTermination()

if __name__ == "__main__":
    # initialize the pipeline with configuration variables
    inference = FraudDetectionInference(config_path = '/app/config.yaml')

    # start streaming process and block until termination
    inference.run_inference()