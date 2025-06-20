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
        "amount": {"type": "number", "minimum": 0.01, "maximum": 10000},
        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "merchant": {"type": "string"},
        "timestamp": {
            "type": "string",
            "format": "date-time"
        },
        "location": {"type": "string", "pattern": "^[A-Z]{2}$"},
        "device_id": {"type": "string"},
        "ip_address": {"type": "string", "format": "ipv4"},
        "new_device_flag": {"type": "integer", "minimum": 0, "maximum": 1},
        "is_fraud": {"type": "integer", "minimum": 0, "maximum": 1},
        'note': {"type": "string"}
    },
    "required": ["transaction_id", "user_id", "amount", "currency", "merchant", "timestamp", "location", "device_id", "ip_address", "new_device_flag", "is_fraud", "note"]
}

class TransactionProducer():
    def __init__(self):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092') # bootstrap server is where the Kafka configuration comes from
        self.kafka_username = os.getenv('KAFKA_USERNAME')
        self.kafka_password = os.getenv('KAFKA_PASSWORD')
        self.topic = os.getenv('KAFKA_TOPIC', 'transactions')
        self.running = False
        self.user_transaction_history = defaultdict(list) # track past transaction times for velocity check
        self.user_devices = defaultdict(lambda: random.sample([fake.uuid4() for _ in range(5)], random.randint(1, 3)))
        self.user_ips = defaultdict(lambda: random.sample([fake.ipv4_public() for _ in range(5)], random.randint(1, 3)))

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
                'security.protocol': 'SASL_SSL',
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
        self.country_list = [
            'US',  # United States
            'CA',  # Canada
            'GB',  # United Kingdom
            'DE',  # Germany
            'FR',  # France
            'AU',  # Australia
            'IN',  # India
            'SG',  # Singapore
            'NL',  # Netherlands
            'SE',  # Sweden
            'JP',  # Japan
            'KR',  # South Korea
            'BR',  # Brazil
            'ZA',  # South Africa
            'MX',  # Mexico
            'IT',  # Italy
            'ES',  # Spain
            'AE',  # United Arab Emirates
            'HK',  # Hong Kong
            'CH'   # China
        ]
        self.user_home_country_map = {
            user_id: random.choice(self.country_list) for user_id in range(1000, 10000)
        }
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
            transaction['note'] = 'High Transaction Velocity anomaly detected'
    
    # function to get the transactions location for an anomaly 
    def get_transaction_location(self, user_id: int, anomaly_chance: float = 0.01) -> str:
        home_country = self.user_home_country_map[user_id]

        if random.random() < anomaly_chance:
            alt_countries = [country for country in self.country_list if country != home_country]
            return random.choice(alt_countries)
        return home_country

    # function to simulate unseen device and IP address 
    def get_device_info(self, user_id: int, anomaly_chance: float = 0.01) -> Dict[str, Any]:
        if random.random() < anomaly_chance:
            # simulate a new device/IP address 
            return {
                'device_id': str(fake.uuid4()),
                'ip_address': fake.ipv4_public(),
                'new_device_flag': 1
            }
        else:
            return {
                'device_id': random.choice(self.user_devices[user_id]),
                'ip_address': random.choice(self.user_ips[user_id]),
                'new_device_flag': 0 
            }

    # function to validate a transaction 
    def validate_transaction(self, transaction) -> bool:
        try:
            validate(
                instance = transaction, 
                schema = TRANSACTION_SCHEMA,
                format_checker = FormatChecker()
            )
            return True
        except ValidationError as e:
            logger.error(f'Invalid transaction: {e.message}')
            logger.debug(f'Transaction content: {json.dumps(transaction, indent=2)}')
            return False

    # function to generate the transaction: returns a dict in a string format of the transaction 
    def generate_transaction(self) -> Optional[Dict[str, Any]]:
        user_id = random.randint(1000, 9999)
        transaction = {
            'transaction_id': fake.uuid4(),
            'user_id': user_id,
            'amount': round(fake.pyfloat(min_value = 0.01, max_value = 10000), 2),
            'currency': 'USD',
            'merchant': fake.company(),
            'timestamp': (datetime.now(timezone.utc) + timedelta(seconds = random.randint(-300, 3000))).isoformat(), # use UTC time so we know there is no discrepancy between transactions that happen in different timezones
            'location': self.get_transaction_location(user_id),
            'is_fraud': 0,
            'note': ''
        }

        is_fraud = 0 
        amount = transaction['amount']
        user_id = transaction['user_id']
        merchant = transaction['merchant']
        device_info = self.get_device_info(user_id)
        transaction.update(device_info)

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
                transaction['note'] = 'Account Takeover anomaly detected'

        # card testing 
        if not is_fraud and amount < 2.0:
            # simulate rapid small transactions 
            if user_id % 1000 == 0 and random.random() < 0.25:
                is_fraud = 1 
                transaction['amount'] = round(random.uniform(0.01, 2), 2)
                transaction['location'] = 'US'
                transaction['note'] = 'Card Testing anomaly detected'

        # merchant collusion 
        if not is_fraud and merchant in self.high_risk_merchants:
            if amount > 3000 and random.random() < 0.15:
                is_fraud = 1 
                transaction['amount'] = round(random.uniform(300, 1500), 2)
                transaction['note'] = 'Merchant Collusion anomaly detected'
          
        # geographic anomalies 
        if not is_fraud and transaction['location'] != self.user_home_country_map[user_id]:
            is_fraud = 1 
            transaction['note'] = 'Geo anomaly detected'

        # flag new device anomalies 
        if not is_fraud and device_info['new_device_flag'] == 1:
            is_fraud = 1 
            transaction['note'] = 'New device anomaly detected'

        # eshtablish the baseline for random fraud (~0.1 - 0.3%)
        if not is_fraud and random.random() < 0.002:
            is_fraud = 1 
            transaction['amount'] = random.uniform(100, 2000)

        # ensure that the final fraud rate is between 1-2% 
        if not is_fraud and random.random() < 0.015:
            is_fraud = 1
        transaction['is_fraud'] = is_fraud

        # validate the modified transaction 
        if self.validate_transaction(transaction):
            return transaction 
        logger.warning("Transaction failed schema validation and was not returned.")
        return None

    # function to send a transaction to Kafka 
    def send_transaction(self) -> bool:
        try:
            transaction = self.generate_transaction()

            if not transaction:
                logger.warning("No transaction returned from generate_transaction()")
                return False 
            
            logger.info(f"Sending transaction: {transaction['transaction_id']}")

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
            logger.info('Producer stopped...')

# entry point
if __name__ == "__main__":
    producer = TransactionProducer()
    producer.run_continuous_production()

