import os
from confluent_kafka import Producer
import logging 
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
        

# entry point
if __name__ == "__main__":
    producer = TransactionProducer()
    producer.run_continuous_production()

