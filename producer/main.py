import os
from confluent_kafka import Producer
import logging 
import random
from dotenv import load_dotenv
from faker import Faker


logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    level = logging.INFO
)

logger = logging.getLogger(__name__)

# load the environment variables needed for Kafka setup 
load_dotenv(dotenv_path = '/app/.env')

# import faker to generate transactions 
fake = Faker()

class TransactionProducer():
    def __init__(self):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092') # bootstrap server is where the Kafka configuration comes from
        self.kafka_username = os.getenv('KAFKA_USERNAME')
        self.kafka_password = os.getenv('KAFKA_PASSWORD')
        self.topic = os.getenv('KAFKA_TOPIC', 'transactions')
        self.running = False

        # confluent kafka configuration
        self.producer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': 'transaction-producer',
            'compression.type': 'gzip',
            'linger.ms': '5',
            'batch.size': 16384
        }

        # if there is a username and password, validate the username and password
        if self.kafka_username and self.kafka_password:
            self.producer_config.update({
                'security_protocol': 'SASL_SSL',
                'sasl.mechanism': 'PLAIN',
                'sasl.username': self.kafka_username,
                'sasl.password': self.kafka_password
            })
        else:
            self.producer_config['security.protocol'] = 'PLAINTEXT'

        try:
            self.producer = Producer(self.producer_config)
            logger.info('Confluent Kafka Producer has been initialized successfully\n')
        except Exception as e:
            logger.error(f'Failed to initialize Confluent Kafka Producer: {str(e)}')
            raise e 
        
        # handle the compromised users/merchants, that we know are risky
        self.compromised_users = set(random.sample(range(1000, 9999), 50)) # set 0.5% of users to be comprimised users (0.5% so the model does not predict comprimised users too often)
        self.high_risk_merchants = ['QuickCash', 'GlobalDigital', 'FastMoneyX']
        self.fraud_pattern_weights = {
            'account_takeover': 0.4, # 40% of fraud cases when someone takes over your account (they get your password for example, as this is the most common for fraud cases)
            'card_testing': 0.3, # 30% of fraud cases when someone tests your card to see if it has money or not
            'merchant_collusion': 0.2, # 20% 
            'geo_anomaly': 0.1 # 10% of fraud cases when someone uses your card from a different location
        }

        # configure graceful shutdown 
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    # create a function to generate transactions 
    def run_continuous_production(self, interval: float = 0.0):
        pass:

    # create the shutdown function 
    def shutdown(self, signum = None, frame = None):
        if self.running:
            logger.info('Initializing shutdown...')
            self.running = False

            if self.producer:
                self.producer.flush(timeout = 30)
                self.producer.close()
            logger.info('Producer stopped...')

# entry point
if __name__ == "__main__":
    producer = TransactionProducer()
    producer.run_continuous_production()

