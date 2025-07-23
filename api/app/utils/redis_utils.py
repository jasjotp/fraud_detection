from redis import Redis
from datetime import datetime, timedelta
from functools import lru_cache
import os 

# list of high-risk merchants
HIGH_RISK_MERCHANTS = ['QuickCash', 'GlobalDigital', 'FastMoneyX']

# function to return redis connection 
@lru_cache(maxsize = 1)
def get_redis() -> Redis:
    return Redis(host = "redis", port = 6379, decode_responses = True)

# helper finction to add a transaction timestamp for a user in Redia and count timestamps in last 24h
def increment_user_activity_24h(user_id: int):
    redis = get_redis()
    key = f"user:{user_id}:activity_timestamps"
    now = datetime.utcnow().timestamp()

    # add current timestamp 
    redis.lpush(key, now)

    # trim list to last 100 items max to avoid bloating in Redis 
    redis.ltrim(key, 0, 99)

    # remove timestamps older than 24h 
    timestamps = redis.lrange(key, 0, -1)
    fresh_timestamps = [float(ts) for ts in timestamps if float(ts) > (now - 86400)]

    # replace with filtered timestamps 
    redis.delete(key)
    for ts in fresh_timestamps:
        redis.rpush(key, ts)
    
    # update the counter key 
    redis.set(f"user:{user_id}:activity24h", len(fresh_timestamps))

# function to get the user activity for the last 24h 
def get_user_activity_24h(user_id: int) -> int:
    redis = get_redis()
    key = "user:{user_id}:activity24h"
    val = redis.get(key)
    return int(val) if val and val.isdigit() else 0 

# function to return the ratio of current amount to average of last 6 transactions (excluding current transaction)
def get_amount_to_avg_ratio(user_id: int, amount: float) -> float:
    redis = get_redis()
    key = f"user:{user_id}:recent_amounts"

    prev_amounts = redis.lrange(key, 0, 5)
    prev_amounts = [float(x) for x in prev_amounts if x.replace('.', '', 1).isdigit()]

    redis.lpush(key, str(amount))
    redis.ltrim(key, 0, 5)

    if not prev_amounts:
        return 1.0

    avg = sum(prev_amounts) / len(prev_amounts)
    return round(amount / avg, 3) if avg != 0 else 1.0

# returns 1 if merchant is high risk else 0 
def get_merchant_risk(merchant: str) -> int:
    if not merchant: 
        return 0 
    return 1 if merchant in HIGH_RISK_MERCHANTS else 0 
