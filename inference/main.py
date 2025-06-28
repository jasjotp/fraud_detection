import os 
import logging
import joblib
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType,
                              IntegerType, DoubleType, TimestampType)
from pyspark.sql.functions import (from_json, col, hour, dayofmonth, pandas_udf, PandasUDFType,
                                  dayofweek, when, lit, coalesce, window, avg, unix_timestamp, lag)
from pyspark.sql.window import Window
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
import yaml
import json

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
            logger.info(f'Model loaded from: {model_path}')
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
        transactions = df.alias('txns')
        activity = user_activity_24h.alias('a')

        combined_df = (
            transactions.join(
                activity, 
                (transactions.user_id == activity.user_id) &
                    (transactions.timestamp >= activity.window.start) &
                    (transactions.timestamp < activity.window.end),
                how = 'left'
            )
            .drop(activity.user_id)
            .drop(activity.window)
        )

        return combined_df

    # function to get the ratio of the current amount to the rolling mean of the last 6 transactions (exluding current one) 
    def amount_to_avg_ratio(self, df):
        '''
        Adds a column amount_to_avg_ratio to the df that is the curent transaction amount dividied by the 
        mean amount of the last 6 transactions (excluding the current one) per user, ordered by timestamp
        to see if the current amount is a large amount compared to past transactions. If the user has no
        transactions, the ratio defaults to 1.0
        '''
        # window that includes the 6 rows before the current one 
        prev_6_transactions = (
            Window
            .partitionBy('user_id')
            .orderBy('timestamp')
            .rowsBetween(-6, -1) # include the 6 rows before the current row up until 1 row before the current row (0 - 1)
        )

        # make a column in the df that contains the avg amount of the last 6 transactions
        df = df.withColumn('avg_amount_prev_6_txns', avg(col('amount')).over(prev_6_transactions))

        # create a column that contains the ratio for the current amount compared to the avg amount of the last 6 transactions
        df = df.withColumn('amount_to_avg_ratio', 
                           col('amount') / col('avg_amount_prev_6_txns')
                           ).fillna({'amount_to_avg_ratio': 1.0})
        return df.drop('avg_amount_prev_6_txns')

    # function to add features into the dataframe 
    def add_features(self, df):
        df = df.withColumn('transaction_hour', hour(col('timestamp'))) # hour the transaction occurred

        df = df.withColumn('is_weekend', 
                        ((dayofweek(col('timestamp')) == 1) | (dayofweek(col('timestamp')) == 7)).cast('int')) # flag whether the tranaction happend on a weekend or not 
        
        df = df.withColumn('is_night',
                           ((hour(col('timestamp')) >= 22) | (hour(col('timestamp')) < 5)).cast('int')) # whehter the transaction happen overnight (between 10PM and 5AM)
        
        df = df.withColumn('transaction_day', dayofweek(col('timestamp')))

        df = self.get_user_activity_24h(df) # counts the number of transacions for each user in a rolling 24 hour window using timestamp
        df = self.amount_to_avg_ratio(df) # get the ratio of the current amount to the rolling mean of the last 6 transactions (exluding current one) 
        
        # create a feaure that flags whether the merchant is a high risk merchant or not 
        high_risk_merchants = self.config.get('high_risk_merchants', ['QuickCash', 'GlobalDigital', 'FastMoneyX'])
        df = df.withColumn('merchant_risk',
                           when(col('merchant').isin(high_risk_merchants), lit(1)).otherwise(lit(0)))

        # feature to find the time elaspsed since the last transaction per user (in seconds)
        win = Window.partitionBy('user_id').orderBy('timestamp')

        # calculate previous timestamp for each user's transaction
        df = df.withColumn('prev_timestamp', lag('timestamp').over(win))

        # subtract the current transactions timestamp with the previous timestamp for the user to find each transaction/user's time since last transaction (in seconds)
        df = df.withColumn('time_since_last_txn',
                           (unix_timestamp('timestamp') - unix_timestamp('prev_timestamp')).cast('double')
                        ).fillna({'time_since_last_txn': 0.0}).drop('prev_timestamp')
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
        logger.info(f"Successfully read data from Kafka topic: {self.config['kafka']['topic']}")
        df = self.add_features(df)
        logger.info("Successfully added features to dataframe")

        broadcast_model = self.broadcast_model
        threshold = self.get_threshold(experiment_name = 'fraud_detection_')

        # create a udf function to predict the value (0/1) of the transaction
        @pandas_udf('int', PandasUDFType.SCALAR)
        def predict_udf(
                user_id: pd.Series, 
                amount: pd.Series, 
                merchant: pd.Series, 
                currency: pd.Series, 
                transaction_hour: pd.Series,
                is_weekend: pd.Series,
                is_night: pd.Series,
                time_since_last_txn: pd.Series,
                transaction_day: pd.Series,
                merchant_risk: pd.Series,
                user_activity_24h: pd.Series,
                amount_to_avg_ratio: pd.Series
        ) -> pd.Series:
            # list out the schema that we want to predict on, as a pandas df
            input_df = pd.DataFrame({
                'user_id': user_id,
                'amount': amount,
                'merchant': merchant,
                'currency': currency,
                'transaction_hour': transaction_hour,
                'is_weekend': is_weekend,
                'is_night': is_night,
                'time_since_last_txn': time_since_last_txn,
                'transaction_day': transaction_day,
                'merchant_risk': merchant_risk,
                'user_activity_24h': user_activity_24h,
                'amount_to_avg_ratio': amount_to_avg_ratio
            })

            # get probabilities of the fraud cases 
            prob = broadcast_model.value.predict_proba(input_df)[:, 1]
            predictions = (prob >= threshold).astype(int)

            return pd.Series(predictions)
        
        # predict on the new transactions that come in
        prediction_df = df.withColumn('prediction',
                                    predict_udf(
                                        col('user_id'),
                                        col('amount'),
                                        col('merchant'),
                                        col('currency'),
                                        col('transaction_hour'),
                                        col('is_weekend'),
                                        col('is_night'),
                                        col('time_since_last_txn'),
                                        col('transaction_day'),
                                        col('merchant_risk'),
                                        col('user_activity_24h'),
                                        col('amount_to_avg_ratio')
                                    ))

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
