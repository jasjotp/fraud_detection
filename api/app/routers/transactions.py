from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession 
from app.database import get_session 
from app.models import Transaction, TxnCreate, Prediction
from app.predict import predict_fraud
from app.kafka_consumer import ensure_user_exists 
from app.utils.redis_utils import (get_user_activity_24h, get_amount_to_avg_ratio, get_merchant_risk, increment_user_activity_24h)
from sqlalchemy.future import select 
from datetime import datetime 

router = APIRouter()

# GET method to return all transactions 
@router.get('/', tags = ['Transactions'])
async def get_all_transactions(limit: int = 1000, offset: int = 0, session: AsyncSession = Depends(get_session)):
    transactions = await session.execute(
        select(Transaction).offset(offset).limit(limit)
    )
    return transactions.scalars().all()

# POST method to post/create a transaction and generate its prediction using loaded model created before
@router.post('/', tags = ['Transactions'])
async def create_transaction(txn: TxnCreate, session: AsyncSession = Depends(get_session)):
    await ensure_user_exists(txn.user_id, session)

    # save the transaction
    new_txn = Transaction(
        transaction_id = txn.transaction_id,
        user_id = txn.user_id,
        amount = txn.amount,
        currency = txn.currency,
        merchant = txn.merchant,
        timestamp = txn.timestamp,
        location = txn.location
    )

    session.add(new_txn)
    await session.commit()
    await session.refresh(new_txn)

    # compute dynamic features 
    try: 
        user_activity_24h = get_user_activity_24h(txn.user_id)
        amount_to_avg_ratio = get_amount_to_avg_ratio(txn.user_id, txn.amount)
        merchant_risk = get_merchant_risk(txn.merchant)

    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code = 500, detail = f'Dynamic feature error: {e}')
    
    features = {
        'transaction_id': txn.transaction_id, 
        'transaction_hour': txn.timestamp.hour,
        'transaction_day': txn.timestamp.weekday(),
        'is_weekend': txn.timestamp.weekday() >= 5, 
        'is_night': txn.timestamp.hour < 6 or txn.timestamp.hour > 22,
        'user_activity_24h': user_activity_24h, 
        'amount_to_avg_ratio': amount_to_avg_ratio, 
        'merchant_risk': merchant_risk,
        'amount': txn.amount, 
        'merchant': txn.merchant or 'unknown'
    }

    # make prediction 
    prediction = predict_fraud(features)
    is_fraud = prediction['is_fraud']
    probability = prediction['probability']

    # save the prediction 
    pred = Prediction(
        transaction_id = txn.transaction_id,
        is_fraud = is_fraud, 
        transaction_hour = features['transaction_hour'], 
        transaction_day = features['transaction_day'],
        is_weekend = features['is_weekend'],
        is_night = features['is_night'],
        user_activity_24h = user_activity_24h,
        amount_to_avg_ratio = amount_to_avg_ratio, 
        merchant_risk = merchant_risk
    )
    session.add(pred)
    await session.commit()

    # update the user activity in the last 24h metric for the user
    increment_user_activity_24h(txn.user_id)

    return {
        "message": "Transaction added successfully", 
        "transaction_id": txn.transaction_id, 
        "prediction": {
            "is_fraud": is_fraud,
            "probability": probability
        }
    }

# DELETE method to delete a transaction and its prediction 
@router.delete('/{transaction_id}', tags = ['Transactions'])
async def delete_transaction(transaction_id: str, session: AsyncSession = Depends(get_session)):
    # delete the prediction first if it exists 
    pred_result = await session.execute(select(Prediction).where(Prediction.transaction_id == transaction_id))

    prediction = pred_result.scalar_one_or_none()

    # if there is a valid prediction (1): delete that prediction first
    if prediction: 
        await session.delete(prediction)
    
    # delete the transaction 
    txn_result = await session.execute(select(Transaction).where(Transaction.transaction_id == transaction_id))

    # if the transaction has only one matching reocrd, delete it, if it does not exist, return None
    transaction = txn_result.scalar_one_or_none()
    if not transaction: 
        raise HTTPException(status_code = 404, detail = 'Transaction not found')
    
    await session.delete(transaction)
    await session.commit()
    
    return {"message": f"Transaction {transaction_id} (and prediction if any) deleted successfully"}
