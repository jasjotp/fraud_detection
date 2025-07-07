import os
import logging 
import boto3
import yaml
import mlflow
import pandas as pd
import numpy as np
import json
import joblib
import optuna
from datetime import datetime
from kafka import KafkaConsumer
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score, average_precision_score, precision_recall_curve, confusion_matrix, f1_score
from xgboost import XGBClassifier
from math import sin, cos, pi
import matplotlib.pyplot as plt

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers = [
        logging.FileHandler('./fraud_detection_model.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# helper function to flag if the transaction's location is different from user's most common location (home location)
def flag_home_location_mismatches(df):
    # get the most common location, using the mode, per user 
    home_locations = df.groupby('user_id')['location'].agg(lambda x: x.mode()[0])

    # map each home location to each row 
    df['home_location'] = df['user_id'].map(home_locations)

    # flag if transaction location != home location 
    df['is_location_mismatch'] = (df['location'] != df['home_location']).astype(int)
    return df 

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

        # extract the sine and cosine of the transaction hour so that the model can learn cyclic nature like hour 23 and 0 actually being close to each other
        df['transaction_hour_sin'] = np.sin(2 * pi * df['transaction_hour'] / 24)
        df['transaction_hour_cos'] = np.cos(2 * pi * df['transaction_hour'] / 24)

        # flag transactions happening at night (between 10PM and 5AM)
        df['is_night'] = ((df['transaction_hour'] >= 22) | df['transaction_hour'] < 5).astype(int)

        # flag transactions happening on the weekend (Saturday = 5, Sunday = 6)
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)

        # flag transaction day 
        df['transaction_day'] = df['timestamp'].dt.day
 
        # flag transaction day of week 
        df['transaction_dayOfWeek'] = df['timestamp'].dt.dayofweek

        # flag high velocity transactions (5 or more in the same minute per user)
        df['minute'] = df['timestamp'].dt.floor('min')

        # count transactions per user per minute 
        df['user_transactions_per_minute'] = df.groupby(['user_id', 'minute'])['timestamp'].transform('count')
        df['user_transactions_per_minute'] = df['user_transactions_per_minute'].fillna(1)

        # flag is transaction count in that minute is 5 or more 
        df['is_high_velocity'] = (df['user_transactions_per_minute'] >= 5).astype(int)

        # extract a feature to find the number of days it has been since the user transacted 
        df['days_since_last_transaction'] = (df['timestamp'] - df.groupby('user_id')['timestamp'].shift()).dt.total_seconds() / 86400
        df['days_since_last_transaction'].fillna(df['days_since_last_transaction'].median(), inplace = True)

        # flag if transactions happen at an hour that is rare for the user 
        # build a user-hour frequency profile 
        user_hour_profile = (
            df.groupby(['user_id', 'transaction_hour'])
            .size()
            .groupby('user_id')
            .apply(lambda x: x / x.sum())
            .droplevel(0) # drops the outer user_id index level
        )
        
        # find the hours where the user rarely transacts at (less than 5% of the time)
        rare_hours = user_hour_profile[user_hour_profile < 0.05].rename('hour_fraction').reset_index()
        rare_hours['is_rare_hour'] = 1

        # merge the rare hours back into the original df 
        df = df.merge(rare_hours[['user_id', 'transaction_hour', 'is_rare_hour']], on = ['user_id', 'transaction_hour'], how = 'left')
        df['is_unusual_hour_for_user'] = df['is_rare_hour'].fillna(0).astype(int)
        df.drop(columns = ['is_rare_hour'], inplace = True)

        # extract behavioural features 
        # get user activity in last 24 hours: a rolling transaction group on the timestamp, to get each users amount/transaction count in the last 24 hours 
        df['user_activity_24h'] = df.groupby('user_id', group_keys = False).apply(
            lambda g: g.rolling('24h', on = 'timestamp', closed = 'left')['amount'].count().fillna(0)
        )

        # find the  time between the current and past transactions per user
        df['time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        df['time_diff'] = df['time_diff'].fillna(0)

        # find the rolling average transaction interval for that user for each transaction (find each users average time between past transactions)
        df['user_avg_transaction_interval'] = (
            df.groupby('user_id')['time_diff']
            .expanding()
            .mean()
            .shift()
            .reset_index(level = 0, drop = True)
        )

        df['user_avg_transaction_interval'] = df['user_avg_transaction_interval'].fillna(df['user_avg_transaction_interval'].median())

        # calculate the zscore amount per user to catch outliers based on each user's transaction history 
        # calcualte each user's rolling mean and std to date using expanding()
        df['zscore_amount_per_user'] = (
            df.groupby('user_id')['amount']
            .apply(lambda x: (x - x.expanding().mean().shift()) / x.expanding().std().shift())
            .reset_index(level = 0, drop = True)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

        # create a feature for burst detection: < $2 transactions in the last minute
        small_txn_count = (
            df[df['amount'] < 2.0]
            .set_index('timestamp')
            .groupby('user_id')['amount']
            .rolling('1min', closed = 'left')
            .count()
            .reset_index()
            .rename(columns = {'amount': 'burst_small_txn_count_last_min'})
        )
        df = df.merge(small_txn_count, on = ['user_id', 'timestamp'], how = 'left')
        df['burst_small_txn_count_last_min'].fillna(0, inplace = True)

        # count of all transactions for each user in the last 5 minutes
        txn_count_5min = (
            df.set_index('timestamp')
            .groupby('user_id')['transaction_id']
            .rolling('5min', closed = 'left')
            .count()
            .reset_index()
            .rename(columns = {'transaction_id': 'txn_count_last_5min'})
        )
        df = df.merge(txn_count_5min, on = ['user_id', 'timestamp'], how = 'left')
        df['txn_count_last_5min'].fillna(0, inplace = True)
        
        # extract monetary features
        
        # extract the last transaction amount, as compared to the avererage amount of the last 6 transactions (not including the current one)
        df['amount_to_avg_ratio'] = (
            df.groupby('user_id')['amount']
            .apply(lambda g: g / g.shift(1).rolling(7, min_periods = 1).mean())
            .reset_index(level = 0, drop = True)
            .fillna(1.0)
        )

        # extract the rolling standard deviation for each user for their last 10 tramsactions
        df['user_transaction_amount_std'] = (
            df.groupby('user_id')['amount']
            .transform(lambda x: x.rolling(window = 10, min_periods = 2).std())
        )
        df['user_transaction_amount_std'].fillna(df['user_transaction_amount_std'].median(), inplace = True)

        # extract the average amount spent by each user in the last 7 days 
        amount_7d_avg = (
            df.set_index('timestamp')
                .groupby('user_id')['amount']
                .rolling('7d', min_periods = 1)
                .mean()
                .reset_index()
                .rename(columns = {'amount': 'amount_7d_avg'})
        )

        # merge the average amount spent by each user in the last 7 days correctly
        df = df.merge(amount_7d_avg, on = ['user_id', 'timestamp'], how = 'left')
        df['amount_7d_avg'].fillna(df['amount_7d_avg'].median(), inplace = True)

        # extract the current transaction amount compared to the average transaction amount for the last 7 days for the user as a ratio 
        df['amount_to_avg_ratio_7d'] = df['amount'] / df['amount_7d_avg']
        df['amount_to_avg_ratio_7d'] = df['amount_to_avg_ratio_7d'].fillna(1.0)

        # extract the current amount compared to the rolling median of the past 5 transactions for the user
        df['amount_vs_median'] = df.groupby('user_id')['amount'].transform(
            lambda x: (x - x.rolling(window = 5, min_periods = 1).median()).abs()
        )
        df['amount_vs_median'] = df['amount_vs_median'].fillna(df['amount_vs_median'].median())

        # total historical spend per user (cumulative) to date (excluding current date)
        df['user_total_spend_todate'] = df.groupby('user_id')['amount'].cumsum().shift().fillna(0)

        # calculate the amount spent for each user in the last 24h, excluding the current transaction
        amount_spent_last24h = (
            df.set_index('timestamp')
                .groupby('user_id')['amount']
                .rolling('24h', closed = 'left')
                .sum()
                .reset_index()
                .rename(columns = {'amount': 'amount_spent_last24h'})
        )

        # merge the average amount spent by each user in the last 24 hours correctly back to main df
        df = df.merge(amount_spent_last24h, on = ['user_id', 'timestamp'], how = 'left')
        df['amount_spent_last24h'] = df['amount_spent_last24h'].fillna(0)

        # extract a ratio for the amount a user spemt in the last 24h compared to their historical total spend to capture binge behaviour like if a user transaction has suddently spiked and spent 60% of their money in the last 24h, that is suspicious
        df['user_spending_ratio_last24h'] = (
            df['amount_spent_last24h'] / df['user_total_spend_todate'].replace(0, np.nan)
        ).fillna(0)

        # find the current amount's ratio compared to the previous amount, as big jumps in amount can signal unusual behaviour
        df['prev_amount'] = df.groupby('user_id')['amount'].shift().fillna(0)

        df['amount_change_ratio'] = (
            (df['amount'] - df['prev_amount']) / df['prev_amount'].replace(0, np.nan)
        ).fillna(0)

        # merchant features 
        high_risk_merchants = self.config.get('high_risk_merchants', ['QuickCash', 'GlobalDigital', 'FastMoneyX'])
        df['merchant_risk'] = df['merchant'].isin(high_risk_merchants).astype(int) # 1 or 0 

        # extract a feature to see how often the user interacts with this merchant 
        df['user_merchant_transaction_count'] = df.groupby(['user_id', 'merchant'])['amount'].transform('count')

        # since fraudsters often test stolen cards across many vendors quickly, count unique merchants for each user_id in a rolling 24 hour window 
        num_distinct_merchants_24h = (
            df.set_index('timestamp')
            .groupby('user_id')['merchant']
            .resample('24h')
            .nunique()
            .reset_index()
            .rename(columns = {'merchant': 'num_distinct_merchants_24h'})
        )
        
        # merge the number of merchants for each user in the last 24 hours correctly back to main df
        df = df.merge(num_distinct_merchants_24h, on = ['user_id', 'timestamp'], how = 'left')
        df['num_distinct_merchants_24h'] = df['num_distinct_merchants_24h'].fillna(0)

        df = df.reset_index() # bring timestamp back as a column

        # location based anomoalies 
        df['prev_location'] = df.groupby('user_id')['location'].shift()
        df['is_location_anomalous'] = (df['location'] != df['prev_location']).astype(int)

        # flag if the user's home location (most common location) is different than the transactions location (is_location_mismatch)
        df = flag_home_location_mismatches(df)
        
        # extract the feature columns we want to use 
        feature_cols = [
            'amount', 
            'is_night', 
            'is_weekend',
            'transaction_day', 
            'user_activity_24h', 
            'amount_to_avg_ratio', 
            'merchant_risk', 
            'merchant',
            'transaction_hour',
            'transaction_hour_sin',
            'transaction_hour_cos',
            'transaction_dayOfWeek',
            'user_transactions_per_minute',
            'is_high_velocity',
            'days_since_last_transaction',
            'is_unusual_hour_for_user',
            'time_diff',
            'user_avg_transaction_interval',
            'zscore_amount_per_user',
            'burst_small_txn_count_last_min',
            'txn_count_last_5min',
            'user_transaction_amount_std',
            'amount_7d_avg',
            'amount_to_avg_ratio_7d',
            'amount_vs_median',
            'user_total_spend_todate',
            'amount_spent_last24h',
            'user_spending_ratio_last24h',
            'prev_amount',
            'amount_change_ratio',
            'user_merchant_transaction_count',
            'num_distinct_merchants_24h',
            'is_location_anomalous',
            'is_location_mismatch'
            ]

        if 'is_fraud' not in df.columns:
            raise ValueError('Missing target column: "is_fraud"')

        return df[feature_cols + ['is_fraud']]

    # create a function to train the model
    def train_model(self):
        try:
            logger.info('Starting model training process...')

            # read in transaction data from Kafka
            df = self.read_from_kafka()

            # feature engineering of our raw data into categorical and numerical variables that our model will use 
            data = self.create_features(df)
            
            # you don't want the model to cheat and already know the is_fraud label, so we remove is_fraud from the input column and have it as our prediction column, so splot data into features (X) and target (Y)
            X = data.drop(columns = ['is_fraud']) # features
            y = data['is_fraud'] # column we want to predict.classify

            # print out all rows that have at least one null 
            null_rows = X.isnull().any(axis = 1)
            rows_giving_error = X[null_rows]
            print(f"{rows_giving_error.shape[0]} rows have NaNs")
            print(rows_giving_error.head())

            # columns that have at least one NaN anywhere
            null_cols = X.columns[X.isnull().any()]
            logger.info(f"Columns with NaNs: {list(null_cols)}")

            # if there are no positive samples, raise an error
            if y.sum() == 0:
                raise ValueError('No positive samples in training data')
            
            if y.sum() < 10:
                logger.warning(f'Low positive samples: {y.sum()} - Consider additional data augmentation')

            # split the data into train and test data (80% of data is used for training and 20% for testing)
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size = self.config['model'].get('test_size', 0.2),
                                                                stratify = y,
                                                                random_state = self.config['model'].get('seed', 42)
            )
            
            # start our MLFLow login 
            with mlflow.start_run():
                mlflow.log_metrics({
                    'train_samples': X_train.shape[0],
                    'positive_samples': int(y_train.sum()),
                    'class_ratio': float(y_train.mean()),
                    'test_samples': X_test.shape[0]
                })
                
                # categorical feature preprocessing
                preprocessor = ColumnTransformer(
                    transformers = [
                        ('merchant_encoder', OrdinalEncoder(
                            handle_unknown = 'use_encoded_value',
                            unknown_value = -1, 
                            dtype = np.float32
                        ), ['merchant'])          
                    ], 
                    remainder = 'passthrough'
                )

                # set the objective function for optuna to optimize feature selection 
                def objective(trial):
                    # boolean flag for every candidate feature
                    feature_flags = {
                        # --- monetary & basic flags ---
                        'amount': trial.suggest_categorical('use_amount', [True, False]),
                        'is_night': trial.suggest_categorical('use_is_night', [True, False]),
                        'is_weekend': trial.suggest_categorical('use_is_weekend', [True, False]),
                        'transaction_day': trial.suggest_categorical('use_transaction_day', [True, False]),
                        'amount_spent_last24h': trial.suggest_categorical('use_amount_spent_last24h', [True, False]),
                        'prev_amount': trial.suggest_categorical('use_prev_amount', [True, False]),

                        # --- activity / velocity ---
                        'user_activity_24h': trial.suggest_categorical('use_user_activity_24h', [True, False]),
                        'user_transactions_per_minute': trial.suggest_categorical('use_user_transactions_per_minute', [True, False]),
                        'is_high_velocity': trial.suggest_categorical('use_is_high_velocity', [True, False]),
                        'txn_count_last_5min': trial.suggest_categorical('use_txn_count_last_5min', [True, False]),

                        # --- ratios & deltas ---
                        'amount_to_avg_ratio': trial.suggest_categorical('use_amount_to_avg_ratio', [True, False]),
                        'amount_to_avg_ratio_7d': trial.suggest_categorical('use_amount_to_avg_ratio_7d', [True, False]),
                        'amount_vs_median': trial.suggest_categorical('use_amount_vs_median', [True, False]),
                        'amount_change_ratio': trial.suggest_categorical('use_amount_change_ratio', [True, False]),
                        'user_spending_ratio_last24h': trial.suggest_categorical('use_user_spending_ratio_last24h', [True, False]),

                        # --- temporal features ---
                        'transaction_hour': trial.suggest_categorical('use_transaction_hour', [True, False]),
                        'transaction_hour_sin': trial.suggest_categorical('use_transaction_hour_sin', [True, False]),
                        'transaction_hour_cos': trial.suggest_categorical('use_transaction_hour_cos', [True, False]),
                        'transaction_dayOfWeek': trial.suggest_categorical('use_transaction_dayOfWeek', [True, False]),
                        'days_since_last_transaction': trial.suggest_categorical('use_days_since_last_transaction', [True, False]),
                        'is_unusual_hour_for_user': trial.suggest_categorical('use_is_unusual_hour_for_user', [True, False]),
                        'time_diff': trial.suggest_categorical('use_time_diff', [True, False]),

                        # --- statistical aggregates ---
                        'user_avg_transaction_interval': trial.suggest_categorical('use_user_avg_transaction_interval', [True, False]),
                        'zscore_amount_per_user': trial.suggest_categorical('use_zscore_amount_per_user', [True, False]),
                        'burst_small_txn_count_last_min': trial.suggest_categorical('use_burst_small_txn_count_last_min', [True, False]),
                        'user_transaction_amount_std': trial.suggest_categorical('use_user_transaction_amount_std', [True, False]),
                        'amount_7d_avg': trial.suggest_categorical('use_amount_7d_avg', [True, False]),
                        'user_total_spend_todate': trial.suggest_categorical('use_user_total_spend_todate', [True, False]),

                        # --- merchant-related ---
                        'merchant_risk': trial.suggest_categorical('use_merchant_risk', [True, False]),
                        'merchant': trial.suggest_categorical('use_merchant', [True, False]),
                        'user_merchant_transaction_count': trial.suggest_categorical('use_user_merchant_transaction_count', [True, False]),
                        'num_distinct_merchants_24h': trial.suggest_categorical('use_num_distinct_merchants_24h', [True, False]),

                        # --- location anomaly flags ---
                        'is_location_anomalous': trial.suggest_categorical('use_is_location_anomalous', [True, False]),
                        'is_location_mismatch':  trial.suggest_categorical('use_is_location_mismatch', [True, False]),
                    }

                    # define the features we want to keep in hte model for sure (the below combination of features gives ~80% precision)
                    base_features = [
                        'amount', 'is_night', 'is_weekend','transaction_day', 
                        'user_activity_24h', 'amount_to_avg_ratio', 'merchant_risk', 'merchant'
                    ]

                    # create the feature subset for the trial 
                    optional_features = [f for f, use_it in feature_flags.items() if use_it]
                    selected_features = list(dict.fromkeys(base_features + optional_features))
                    X_selected = X_train[selected_features]

                    # define hyperparamters for training
                    params = {
                        'max_depth': trial.suggest_int('max_depth', 3, 7),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log = True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'gamma': trial.suggest_float('gamma', 0, 0.3),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5), # L1 regularization weight
                        'n_estimators': self.config['model']['params']['n_estimators'],
                        'eval_metric': 'aucpr',
                        'tree_method': self.config['model'].get('tree_method', 'hist'),
                        'random_state': self.config['model'].get('seed', 42),
                        'n_jobs': -1
                    }                    
                    
                    # XGBoost configuration with optimizations
                    xgb = XGBClassifier(**params)

                    # preprocessing pipeline to train the model (using imbpipeline as we are handling an imbalacned dataset)
                    pipeline = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('smote', SMOTE(random_state = self.config['model'].get('seed', 42))), # address the class imbalance by using boosting methods (SMOTE) so the model can recognize both classes
                        ('classifier', xgb) # XGBoost classififer for prediction
                    ], memory = './cache') # put the pipeline in cache so computation is faster

                    # optimizing for F-beta score (beta=2 emphasizes recall)
                    scores = cross_val_score(
                        pipeline,
                        X_selected,
                        y_train,
                        scoring = make_scorer(fbeta_score, beta = 2, zero_division = 0), # prioritize recall with beta = 2, as we care more about catching false negatives (missing a fradulent transction) instead of catching a few extra false positives (non-fradulent transactioons flagged as fraudulent)
                        cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42),
                        n_jobs = -1
                    )
                    
                    # return mean cross-validation F-beta score
                    return np.mean(scores)

                # conduct hyperparamter tuning and optimization using Optuna 
                study = optuna.create_study(direction = 'maximize')
                logger.info('Starting Optuna hyperparameter optimization...')
                study.optimize(objective, n_trials = 20, show_progress_bar = True)

                logger.info(f'Best hyperparameters using Optuna: {study.best_params}')
                mlflow.log_params(study.best_params)

                # define the features we want to keep in the model for sure (the below combination of features gives ~80% precision)
                base_features = [
                        'amount', 'is_night', 'is_weekend','transaction_day', 
                        'user_activity_24h', 'amount_to_avg_ratio', 'merchant_risk', 'merchant'
                ]

                # reconstruct the best feature list from the best trial 
                best_feature_flags = {f[4:]: use_it 
                                      for f, use_it in study.best_trial.params.items()
                                      if f.startswith('use_') and use_it}
                
                # get the list of best features (keys where value is True)
                best_features = list(dict.fromkeys(base_features + list(best_feature_flags.keys())))
                
                # log the selected features to MLFLow 
                mlflow.log_param('selected_features: ', ', '.join(best_features))

                # find the best model in the pipeline 
                best_model = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state = self.config['model'].get('seed', 42))),
                    ('classifier', XGBClassifier(
                        **study.best_params,
                        eval_metric = 'aucpr',
                        tree_method = self.config['model'].get('tree_method', 'hist'),
                        random_state = self.config['model'].get('seed', 42),
                        n_jobs = -1
                    ))
                ])
                
                # train the model on the best features that optuna selected 
                X_train_selected = X_train[best_features]
                best_model.fit(X_train_selected, y_train)

                # retrieve feature names after the preprocessor has learned encodings
                feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()

                # get importances from the fitted XGB classifier
                importances = best_model.named_steps['classifier'].feature_importances_

                feature_importances_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values(by = 'importance', ascending = False)

                # save the feature importances to a csv and log to MLFlow 
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                feature_importances_path = f'feature_importances_{timestamp}.csv'
                feature_importances_df.to_csv(feature_importances_path, index = False)
                mlflow.log_artifact(feature_importances_path)

                # threshold optimization using the training data
                train_proba = best_model.predict_proba(X_train_selected)[:, 1]

                # get the precision and recall
                precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_train, train_proba)

                # calculcate the F1 score for each of the thresholds
                f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in
                             zip(precision_arr[:-1], recall_arr[:-1])]

                # find the most optimal threshold by getting the max f1 score
                best_threshold = thresholds_arr[np.argmax(f1_scores)]
                logger.info(f'Optimal threhsold determined: {best_threshold:.4f}')

                # model evaluation
                X_test_selected = X_test[best_features]
                X_test_processed = best_model.named_steps['preprocessor'].transform(X_test_selected)

                # probability prediction on test set from the classifier 
                test_proba = best_model.named_steps['classifier'].predict_proba(X_test_processed)[:, 1]

                # apply the optimized threshold to get the biary prediction of whether the transaction is fraud or not 
                y_pred = (test_proba >= best_threshold).astype(int)

                # log our metrics: AUC, precision, recall, f1score, threshold score
                metrics = {
                    'auc_pr': float(average_precision_score(y_test, test_proba)),
                    'precision': float(precision_score(y_test, y_pred, zero_division = 0)),
                    'recall': float(recall_score(y_test, y_pred, zero_division = 0)),
                    'f1': float(f1_score(y_test, y_pred, zero_division = 0)),
                    'threshold': float(best_threshold)
                }

                # log the metrics in MLFlow so we can see the performance of our model
                mlflow.log_metrics(metrics)

                # plot the confusion matrix 
                cm = confusion_matrix(y_test, y_pred)

                plt.figure(figsize = (10, 6))
                plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()

                tick_marks = np.arange(2)
                plt.xticks(tick_marks, labels = ['Not Fraud (0)', 'Fraud (1)'])
                plt.yticks(tick_marks, labels = ['Not Fraud (0)', 'Fraud (1)'])
                plt.xlabel('Predicted Label')
                plt.ylabel('Actual Label')

                labels = [['TN', 'FP'], ['FN', 'TP']]

                for i in range(2):
                    for j in range(2):
                        # TN, FN, FP, TP is the order of confusion_matrix() function
                        plt.text(j, i, 
                                f'{labels[i][j]}\n{cm[i, j]}', 
                                ha = 'center', 
                                va = 'center', 
                                color = 'red',
                                fontsize = 12)
                plt.tight_layout()
                cm_filename = 'confusion_matrix.png'
                plt.savefig(cm_filename)
                mlflow.log_artifact(cm_filename) # log confusion matrix to MLFlow
                plt.close()
                
                # plot the precision and recall curve
                plt.figure(figsize = (10, 6))
                plt.plot(recall_arr, precision_arr, marker = '.', label = 'Precision-Recall Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                pr_curve_filename = 'precision_recall_curve.png'
                plt.savefig(pr_curve_filename)
                mlflow.log_artifact(pr_curve_filename) # log precision-recall curve to MLFlow
                plt.close()

                # log the best model in MLFlow
                signature = infer_signature(X_train_selected, y_pred)

                mlflow.sklearn.log_model(
                    sk_model = best_model, 
                    artifact_path = 'model', 
                    signature = signature,
                    registered_model_name = 'fraud_detection_model'
                )

                # log the model locally 
                os.makedirs('/app/models', exist_ok = True)

                best_model.selected_features_ = best_features
                joblib.dump(best_model, '/app/models/fraud_detection_model.pkl')

                logger.info(f'Training successfully completed with metrics: {metrics}')

                return best_model, metrics

        except Exception as e:
            logger.error(f'Training failed: {e}', exc_info = True)
            raise 