import json 
import logging
from datetime import datetime 
from kafka import KafkaConsumer
from sqlalchemy.exc import IntegrityError 
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio 
from app.database import async_session_factory
from app.models import Transaction, Prediction, User, Role
from faker import Faker
import os 
import yaml
import random

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# function to load config for Kafka 
def load_config(config_path):
    try:
        with open(config_path, 'rb') as f:
            raw = f.read()
            expanded = os.path.expandvars(raw) # allows ${VARIABLE NAME} in config.YAML to get replaced with the actual value from .env
            return yaml.safe_load(expanded)
    except Exception as e:
        logger.error(f'Error loading the config file: {e}')
        raise 

# set config variables to connect to conflent Kafka
def parse_kafka_message(value: str) -> dict:
    return json.loads(value)

# function that enforces the foreign key constraint on transaction, as each transaction must have a valid user id from the user table 
fake = Faker()

async def ensure_user_exists(user_id: int, db: AsyncSession):
    user = await db.get(User, user_id)

    if not user: 
        random_role = random.choice(list(Role))
        user = User(
            id = user_id, 
            first_name = fake.first_name(),
            last_name = fake.last_name(),
            role = random_role
        )

        db.add(user)
        await db.commit()

# function to save only the transaction in the database
async def save_transaction(msg: dict):
    async with async_session_factory() as session:
        try:
            transaction = Transaction(
                transaction_id = msg['transaction_id'],
                user_id = msg['user_id'],
                amount = msg['amount'],
                currency = msg.get('currency'),
                merchant = msg.get('merchant'),
                location = msg.get('location'),
                timestamp = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
            )
            await ensure_user_exists(msg["user_id"], session)
            session.add(transaction)
            await session.commit()
            print(f"Saved transaction for {msg['transaction_id']}")
        except IntegrityError as e:
            await session.rollback()
            if 'unique constraint' in str(e).lower():
                print(f"Duplicate transaction_id: {msg['transaction_id']} - skipping")
            elif 'foreign key constraint' in str(e).lower():
                print(f"Missing user_id {msg['user_id']} - cannot insert transaction {msg['transaction_id']}")
            else:
                print(f"IntegrityError: {e}")
        except Exception as e:
            await session.rollback()
            print(f'Error saving transaction: {e}')

# function to save only the prediction in the database
async def save_prediction(msg: dict):
    async with async_session_factory() as session: 
        try:
            # define prediction Schema 
            prediction = Prediction(
                transaction_id = msg['transaction_id'],
                is_fraud = msg['prediction'],
                transaction_hour = msg['transaction_hour'],
                is_weekend = msg['is_weekend'],
                is_night = msg['is_night'],
                transaction_day = msg['transaction_day'],
                user_activity_24h = msg['user_activity_24h'],
                amount_to_avg_ratio = float(msg['amount_to_avg_ratio']),
                merchant_risk = msg['merchant_risk']
            )
            session.add(prediction)
            await session.commit()
            print(f"Saved prediction for: {msg['transaction_id']}")
        except IntegrityError:
            await session.rollback()
            print(f"Duplicate transaction_id: {msg['transaction_id']} - skipping")
        except Exception as e:
            await session.rollback()
            print(f'Error saving transaction: {e}')

# function to consume transactions from Kafka 
def consume_kafka():
    config = load_config('/app/config.yaml')
    print(f"Loaded Kafka config: {config}")
    try:
        logger.info(" Starting Kafka consumer for streaming to Postgres...")

        consumer = KafkaConsumer(
            'transactions', 'fraud_predictions',  # both topics we want to consume from
            bootstrap_servers = config['kafka']['bootstrap_servers'].split(','),
            security_protocol = 'SASL_SSL',
            sasl_mechanism = 'PLAIN',
            sasl_plain_username = config['kafka']['username'],
            sasl_plain_password = config['kafka']['password'],
            value_deserializer = lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset = 'earliest',
            enable_auto_commit = True,
            group_id = 'fraud-db-writer-02'
        )

        print(f"KafkaConsumer connected. Listening to topics...")

        loop = asyncio.get_event_loop()
        for record in consumer:
            # if the record topic is a transaction or fraud prediction, save it in the apppropriate database 
            msg = record.value 
            print(f"\nReceived message from topic: {record.topic}")
            print(f"Message content: {json.dumps(msg, indent = 2)}") 
            if record.topic == 'transactions':
                loop.run_until_complete(save_transaction(msg))
            elif record.topic == 'fraud_predictions':
                loop.run_until_complete(save_prediction(msg))

    except Exception as e: 
        logger.error(f'Error in Kafka consumer: {e}')

if __name__ == '__main__':
    consume_kafka()