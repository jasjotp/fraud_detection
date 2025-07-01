from functools import lru_cache
from redis import Redis
from pyspark.sql.functions import (pandas_udf, PandasUDFType, udf)
from pyspark.sql.types import LongType, DoubleType

# function to connect to Redis lazily, as we cache the connection 
@lru_cache(maxsize = 1)
def get_redis_connection():
    return Redis(host = 'redis', port = 6379, decode_responses = True)

# function that is a UDF that reads the latest value (user activity per user in the last 24 hours)
@udf(returnType = LongType())
def redis_activity_udf(user_id):
    '''
    uses a per row lookup and the cached Redis connection to get the user id and their number of transactions in last 24 hours and returns a 0 if the key is not present
    '''
    redis = get_redis_connection()
    val = redis.get(f'user:{user_id}:activity24h')
    return int(val) if val is not None else 0

# function that is a UDF and computes the average of the last 6 transactions, and returns the ratio of current amount to average of last 6 transactions (excluding current transaction)
@udf(returnType = DoubleType())
def get_amount_to_avg_ratio_redis(user_id, amount):
    redis = get_redis_connection()
    key = f'user:{user_id}:recent_amounts'

    # fetch the previous 6 amounts (excluding current amount)
    prev_6_amounts = redis.lrange(key, 0, 5)
    prev_6_amounts = [float (x) for x in prev_6_amounts if x.replace('.', '', 1).isdigit()]

    # push the current amount to the start of the list 
    redis.lpush(key, str(amount))

    # trim the list to only keep the latest 6 transactions 
    redis.ltrim(key, 0, 5)

    if len(prev_6_amounts) == 0:
        return 1.0 # default ratio if a user has no previous transactions 

    avg = sum(prev_6_amounts) / len(prev_6_amounts)
    return float(amount) / avg if avg != 0 else 1.0

# UDF function that pulls the time since the last transaction from Redis 
@udf(returnType = DoubleType())
def time_since_last_txn_redis(user_id, curr_ts):
    r = get_redis_connection()
    prev = r.get(f'user:{user_id}:last_ts')
    r.set(f'user:{user_id}:last_ts', str(curr_ts.timestamp()))
    if prev is None:
        return 0.0 
    
    return curr_ts.timestamp() - float(prev)
