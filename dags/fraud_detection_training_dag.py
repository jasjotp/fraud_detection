from airflow import DAG 
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowException
from datetime import datetime, timedelta
import logging 

logging.basicConfig(level = logging.INFO) 
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'Jasjot Parmar',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 16),
    'execution_timeout': timedelta(minutes = 120),
    'max_active_runs': 1
}

# function to train the model 
def train_model(**context):
    '''Airflow wrapper for training task'''    
    try:
        logger.info('Initializing fraud detection training...')
        # call the fraud detection training class to train the model 
        from fraud_detection_training import FraudDetectionTraining
        logger.info('Imported FraudDetectionTraining successfully.')

        trainer = FraudDetectionTraining()
        logger.info('Trainer initialized successfully.')

        # the trainer will return the model and precision 
        model, precision = trainer.train_model()

        return {'status': 'success', 'precision': precision}

    except Exception as e:
        logger.error(f'Training failed: {str(e)}', exc_info = True)
        raise AirflowException(f'Model training failed: {str(e)}')

with DAG(
    dag_id = 'fraud_detection_training',
    default_args = default_args,
    description = 'Fraud detection model training pipeline',
    schedule_interval = '0 3 * * *', # run at 3AM every day
    catchup = False,
    tags = ['fraud', 'machine learning']
) as dag:
    validate_environment = BashOperator( # to validate environment to see if the cluster that is spun up is ready 
        task_id = 'validate_environment',
        bash_command = ''' 
        echo "Validating environment..."
        test -f /app/config.yaml &&
        test -f /app/.env &&
        echo "Environment is valid!"
        '''
    )

    # task to train the model
    training_task = PythonOperator(
        task_id = 'execute_training',
        python_callable = train_model,
        provide_context = True

    )

    # task to clean up resources and environment
    cleanup_task = BashOperator(
        task_id = 'cleanup_resources',
        bash_command = 'rm -f /app/tmp/*.pkl',
        trigger_rule = 'all_done'
    )

    validate_environment >> training_task >> cleanup_task

    # documentation for the DAG
    dag.doc_md = '''
    ## Fraud Detection Training Pipeline 

    Daily training of fraud detection model using: 
        - Transaction data from Kafka
        - XGBoost classification with precision optimization
        - MLFLow for experiment tracking 
    
    '''
