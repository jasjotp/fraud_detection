import os
from confluent_kafka import Producer
import logging 
import random
import json
from dotenv import load_dotenv
from faker import Faker
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import signal
from typing import Optional, Dict, Any
from datetime import timezone
from jsonschema import validate, ValidationError, FormatChecker
import time

logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    level = logging.INFO
)

logger = logging.getLogger(__name__)

# load the environment variables needed for Kafka setup 
load_dotenv(dotenv_path = '/app/.env')

# import faker to generate transactions 
fake = Faker()

# JSON Schema for transaction validation
TRANSACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction_id": {"type": "string"},
        "user_id": {"type": "number", "minimum": 1000, "maximum": 9999},
        "amount": {"type": "number", "minimum": 0.01, "maximum": 100000},
        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "merchant": {"type": "string"},
        "timestamp": {
            "type": "string",
            "format": "date-time"
        },
        "location": {"type": "string", "pattern": "^[A-Z]{2}$"},
        "is_fraud": {"type": "integer", "minimum": 0, "maximum": 1}
    },
    "required": ["transaction_id", "user_id", "amount", "currency", "timestamp", 'is_fraud']
}

class TransactionProducer():
    def __init__(self):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092') # bootstrap server is where the Kafka configuration comes from
        self.kafka_username = os.getenv('KAFKA_USERNAME')
        self.kafka_password = os.getenv('KAFKA_PASSWORD')
        self.topic = os.getenv('KAFKA_TOPIC', 'transactions')
        self.running = False
        self.user_transaction_history = defaultdict(list) # track past transaction times for velocity check

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
    
    # function to deliver report 
    def delivery_report(self, err, msg):
        if err is not None:
            logger.error(f'Message Delivery Failed!: {err}')
        else:
            logger.info(f'Message delivered to: {msg.topic()} [{msg.partition()}]')

    # function to apply a velocity check, to see if many transactions happen in a short time span
    def apply_velocity_check(self, transaction):
        user_id = transaction['user_id']
        transaction_time = datetime.fromisoformat(transaction['timestamp'])
        self.user_transaction_history[user_id].append(transaction_time)

        # keep only transactions in the last 60 seconds 
        window_start = transaction_time - timedelta(seconds = 60)
        recent_times = [t for t in self.user_transaction_history[user_id] if t > window_start]

        # update the user's transaction history with the filtered recent times 
        self.user_transaction_history[user_id] = recent_times

        # if the user has 5 or more transactions in the last minute, then set the transaction to fraud 
        if len(recent_times) >= 5:
            transaction['is_fraud'] = 1 
            transaction['note'] = 'Velocity pattern detected'

    # function to validate a transaction 
    def validate_transaction(self, transaction) -> bool:
        try:
            validate(
                instance = transaction, 
                schema = TRANSACTION_SCHEMA,
                format_checker = FormatChecker()
            )
        except ValidationError as e:
            logger.error(f'Invalid transaction: {e.message}')

    # function to generate the transaction: returns a dict in a string format of the transaction 
    def generate_transaction(self) --> Optional[Dict[str, Any]]:
        transaction = {
            'transaction_id': fake.uuid4(),
            'user_id': random.randint(1000, 9999),
            'amount': round(fake.pyfloat(min_value = 0.01, max_value = 10000), 2),
            'currency': 'USD',
            'merchant': fake.company(),
            'timestamp': (datetime.now(timezone.utc) + timedelta(seconds = random.randint(-300, 3000))).isoformat(), # use UTC time so we know there is no discrepancy between transactions that happen in different timezones
            'location': fake.country_code(),
            'is_fraud': 0
        }

        is_fraud = 0 
        amount = transaction['amount']
        user_id = transaction['user_id']
        merchant = transaction['merchant']

        # apply patterns to generate the fradulent transaction and make sure the transaction is set to fraudulent (is_fraud = 1)

        # simulate velocity checks: if many transactions happen in a short time span 
        if not is_fraud:
            self.apply_velocity_check(transaction)
            is_fraud = transaction['is_fraud']

        # account takeover
        if user_id in self.compromised_users and amount > 500: 
            if random.random() < 0.3:  # 30% chance of fraud in compromized accounts
                is_fraud = 1 
                transaction['amount'] = random.uniform(500, 5000)
                transaction['merchant'] = random.choice(self.high_risk_merchants)

        # card testing 
        if not is_fraud and amount < 2.0:
            # simulate rapid small transactions 
            if user_id % 1000 == 0 and random.random() < 0.25:
                is_fraud = 1 
                transaction['amount'] = round(random.uniform(0.01, 2), 2)
                transaction['location'] = 'US'

        # merchant collusion 
        if not is_fraud and merchant in self.high_risk_merchants:
            if amount > 3000 and random.random() < 0.15:
                is_fraud = 1 
                transaction['amount'] = round(random.uniform(300, 1500), 2)
          
        # geographic anomalies 
        if not is_fraud:
            if user_id % 500 == 0 and random.random() < 0.1:
                is_fraud = 1 
                transaction['location'] = random.choice(['CHINA', 'RUSSIA', 'SAUDI ARABIA', 'PAKISTAN', 'INDIA'])

        # eshtablish the baseline for random fraud (~0.1 - 0.3%)
        if not is_fraud and random.random() < 0.002:
            is_fraud = 1 
            transaction['amount'] = random.uniform(100, 2000)

        # ensure that the final fraud rate is between 1-2% 
        transaction['is_fraud'] = is_fraud if random.random() < 0.985 else 0 

        # validate the modified transaction 
        if self.validate_transaction(transaction):
            return transaction 


    # function to send a transaction to Kafka 
    def send_transaction(self) --> bool:
        try:
            transaction = self.generate_transaction()

            if not transaction:
                return False 
            
            self.producer.produce(
                self.topic,
                key = transaction['transaction_id'],
                value = json.dumps(transaction),
                callback = self.delivery_report
            )

            self.producer.poll(0) # trigger callbacks 
            return True 
        except Exception as e:
            logger.error(f'Error producing message: {e}')
            return False 


    # create a function to generate transactions 
    def run_continuous_production(self, interval: float = 0.0):
        '''
        Run continuous message production with graceful shutdown
        '''
        self.running = True
        logger.info('Starting producer for topic %s...', self.topic)
        
        try:
            while self.running:
                if self.send_transaction():
                    time.sleep(interval)
        finally:
            self.shutdown()

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

