from functools import lru_cache
from redis import Redis
from pyspark.sql.functions import (pandas_udf, PandasUDFType, udf, lit, col, hour)
from pyspark.sql.types import LongType, DoubleType, IntegerType, StringType
import math 
import os 
import json 
import time

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

# UDF function that returns the number of days since the user's last transaction from Redis (defaults to 0.0 for the first transaction) 
@udf(returnType = DoubleType())
def time_since_last_txn_redis(user_id, curr_ts):
    r = get_redis_connection()

    # fetch the previous timestamp from redis for the user
    prev = r.get(f'user:{user_id}:last_ts')

    # update Redis with the current timestamp 
    r.set(f'user:{user_id}:last_ts', str(curr_ts.timestamp()))
    if prev is None:
        return 0.0 
    
    seconds = curr_ts.timestamp() - float(prev)
    return seconds / 86_400.0 # convert to days from seconds (float)

# udf function that increments and gets how many transaction each user had each minute 
@udf('int')
def increment_and_get_minute_count(user_id, timestamp):
    r = get_redis_connection()
    key = f"user:{user_id}:minute:{timestamp.strftime('%Y%m%d%H%M')}"
    count = r.incr(key)

    # set time to live (TTL) so the key vanishes after 90 secs to keep Redis storage clean
    r.expire(key, 90)
    return count

# udf that flags unusual hours for customers (hours where the user trasacts at < 5% of the time)
@udf('int')
def flag_unusual_hour(user_id, timestamp): # returns 1 for unusual and 0 for usual 
    # increment user id: hour_total and user id: hour and then compute the fraction (hour count for that hour / total)
    r = get_redis_connection()
    hr = timestamp.hour
    pipe = r.pipeline()

    total_key = f'user:{user_id}:hour_total'
    hour_key = f'user:{user_id}:hour:{hr:02}'

    # increment the total key and the hour key to store the tootal transaction and hourly transaction count
    pipe.incr(total_key)
    pipe.incr(hour_key)

    total, hour_count, *_ = pipe.execute()
    
    # find the fraction of transactions that happen in that hour (avoid division by 0)
    frac = hour_count / total if total else 1.0 
    return int(frac < 0.05)

# UDF to update and get avg time difference for each user (exclduing the current transaction)
@udf(DoubleType())
def update_and_get_avg_interval(user_id, time_diff):
    if time_diff is None:
        return 0.0
     
    r = get_redis_connection()

    # keys to store cumulative time difference and count 
    sum_key = f'user:{user_id}:interval_sum'
    count_key = f'user:{user_id}:interval_count'

    # get current values 
    pipe = r.pipeline()
    pipe.get(sum_key)
    pipe.get(count_key)
    prev_sum, prev_count = pipe.execute()

    # parse the p43vious sum and previous count values 
    prev_sum = float(prev_sum) if prev_sum is not None else 0.0 
    prev_count = int(prev_count) if prev_count is not None else 0

    # find the avg interval before updating with the current time_diff so we get the previous avg time interval 
    avg_interval = prev_sum / prev_count if prev_count > 0 else 0.0

    # uppdate the pipeline with the current time_diff and increment the 
    pipe = r.pipeline()
    pipe.incrbyfloat(sum_key, float(time_diff))
    pipe.incr(count_key)
    pipe.execute()

    return avg_interval 

# UDF that returns the z-score of amount against that user's previous mean amount, then updates the state
@udf(DoubleType())
def zscore_amount_udf(user_id, amount):
    if amount is None:
        return 0.0 
    
    r = get_redis_connection()

    # keys that hold the running count, sum, and sun of squared amounts
    count_key = f'user:{user_id}:cnt'
    sum_key = f'user:{user_id}:sum'
    sumsq_key = f"user:{user_id}:sumsq"

    # read the previous values 
    pipe = r.pipeline()
    pipe.get(count_key)
    pipe.get(sum_key)
    pipe.get(sumsq_key)
    prev_cnt, prev_sum, prev_sumsq = pipe.execute()

    prev_cnt = int(prev_cnt) if prev_cnt is not None else 0 
    prev_sum = float(prev_sum) if prev_sum is not None else 0.0
    prev_sumsq = float(prev_sumsq) if prev_sumsq is not None else 0.0

    # compute z-score before update to simulate the rolling z score of previous amounts of user
    if prev_cnt >= 2:
        prev_mean = prev_sum / prev_cnt

        # use the population variance formula using running sums 
        prev_var = (prev_sumsq - (prev_sum**2) / prev_cnt) / (prev_cnt - 1)
        prev_std = math.sqrt(prev_var) if prev_var > 0 else 0.0
        zscore = (amount - prev_mean) / prev_std if prev_std > 0 else 0.0
    else:
        zscore = 0.0 # for the first or second transaction, z score is typically defined as 0

    # update running state, including current amount
    pipe = r.pipeline()
    pipe.incr(count_key)
    pipe.incrbyfloat(sum_key, amount)
    pipe.incrbyfloat(sumsq_key, amount * amount)
    pipe.execute()

    # return the z-score
    return float(zscore)

# UDF for rolling stddev of last 10 transaction amounts
@udf(DoubleType())
def rolling_stddev_amount_redis(user_id, amount):
    r = get_redis_connection()
    key = f'user:{user_id}:last10_amounts'

    # get previous amounts 
    prev_amounts = r.lrange(key, 0, 9)
    prev_amounts = [float(x) for x in prev_amounts if x.replace('.', '', 1).isdigit()]

    # update Redlis list: push current amount and trim 
    r.lpush(key, str(amount))
    r.ltrim(key, 0, 9)

    if len(prev_amounts) < 2:
        return 0.0

    mean = sum(prev_amounts) / len(prev_amounts)
    variance = sum((x - mean) ** 2 for x in prev_amounts) / (len(prev_amounts) - 1)
    stddev = math.sqrt(variance)

    return stddev

# UDF that returns the abosolute differnece between the current amount and the rolling median of the past 5 amounts for that user (for the amount_vs_median feature)
@udf(DoubleType())
def amount_vs_median_redis(user_id, amount):
    r = get_redis_connection()
    key = f'user:{user_id}:last5_amounts'

    # pull previous amounts (most recent-first list)
    prev_vals = r.lrange(key, 0, 4)
    prev_vals = [float(x) for x in prev_vals if x]

    # if we have a history of past transactions for that user, compute the median 
    if prev_vals:
        ordered = sorted(prev_vals)
        n = len(ordered)
        mid = n // 2 
        median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
        diff = abs(amount - median)
    else:
        diff = 0.0 # default to 0 for the first transaction for the user

    # push current amount and trim list to five elements 
    r.lpush(key, str(amount))
    r.ltrim(key, 0, 4)

    return diff

# UDF that calculates the the users total spent to date up until until the current transaction 
@udf(DoubleType())
def user_total_spend_todate_redis(user_id, amount):
    r = get_redis_connection()
    key = f'user:{user_id}:total_spend'

    prev_total = r.get(key)
    prev_total = float(prev_total) if prev_total else 0.0

    # store not total including the current amount 
    r.set(key, str(prev_total + amount))

    return prev_total 

# UDF that calculates the the users amount spent in the last 24 hours, excluding until the current transaction 
@udf(DoubleType())
def amount_spent_last24h_redis(user_id, curr_ts, amount):
    r = get_redis_connection()
    key = f'user:{user_id}:txns_24h'
    now_ts = curr_ts.timestamp()

    window_secs = 86400

    # get current window of (timestamp, amount)
    prev = r.lrange(key, 0, -1)
    new_list = []
    total = 0.0

    for record in prev:
        try:
            ts, amt = record.split(':')
            ts = float(ts)
            amt = float(amt)
            if ts >= now_ts - window_secs: # if timestamp is in last 24 hours, add to the total and list 
                total += amt 
                new_list.append(record)
        except:
            continue
    
    # update Redis list by pushing current traansaction and keeping filtreed ones
    new_list.insert(0, f"{now_ts}:{amount}")
    pipe = r.pipeline()
    pipe.delete(key)
    if new_list:
        pipe.rpush(key, *new_list)
        pipe.expire(key, window_secs + 60)
    pipe.execute()

    return total 

# UDF that returns the previous amount the user spent 
@udf(DoubleType())
def prev_amount_redis(user_id, amount):
    r = get_redis_connection()
    key = f"user:{user_id}:prev_amount"

    prev = r.get(key)
    r.set(key, str(amount))
    return float(prev) if prev is not None else 0.0

# UDF that returns how many times the user has interacted with a merchant (excluding current transaction)
@udf(IntegerType())
def user_merchant_txn_count_redis(user_id, merchant):
    r = get_redis_connection()

    key = f'user:{user_id}:merchant:{merchant}:count'
    prev = r.get(key) # get the count so far 
    r.incr(key) # increment the counts

    return int(prev) if prev is not None else 0 

# UDF that computes a rolling 24 hour count of unique merchants per user, excluding the current transaction
@udf(IntegerType())
def num_distinct_merchants_24h_redis(user_id, merchant, ts):
    r = get_redis_connection()
    key = f"user:{user_id}:merchants24h"

    window_secs = 86_400
    now = ts.timestamp()

    # drop entries older than 24h
    r.zremrangebyscore(key, '-inf', now - window_secs)

    # count distinct merchants before the current transaction 
    prev_count = r.zcard(key)

    # update the current merchant so its in the window next time 
    r.zadd(key, {merchant: now})

    return int(prev_count)

# UDF that returns the previous location seen for the user id 
@udf(StringType())
def prev_location_redis(user_id, location):
    r = get_redis_connection()
    key = f'user:{user_id}:prev_location'
    prev_loc = r.get(key)
    r.set(key, location)

    return prev_loc or ''

# UDF that returns 1 if the location differs from the users current 'home' location (the most common location of that user) and 0 otherwise
@udf(IntegerType())
def is_location_mismatch_redis(user_id, location):
    r = get_redis_connection()

    # set the keys in Redis 
    home_key = f'user:{user_id}:home_location'
    count_key = f'user:{user_id}:loc_counts'

    # read the current home location 
    home_loc = r.get(home_key)

    # set the mismatch flag (1 if the location differs from the home location)
    mismatch = int(home_loc is not None and location != home_loc)

    # update count for this location 
    new_count = r.hincrby(count_key, location, 1)

    # if the home location is empty, set the first ever location in the transaction to be home location 
    if home_loc is None:
        r.set(home_key, location)
    else:
        # else, comparet the counts of locations and set a new home location if any new locations count if greater than the current home location count
        current_home_count = int(r.hget(count_key, home_loc) or 0)
        if new_count > current_home_count:
            r.set(home_key, location)

    return mismatch
