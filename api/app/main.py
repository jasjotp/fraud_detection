from fastapi import FastAPI 
from app.routers import transactions, users, predict 

app = FastAPI(title = 'Fraud Detection API')

app.include_router(transactions.router, prefix = '/transactions')
app.include_router(users.router, prefix = '/users')
app.include_router(predict.router, prefix = '/predict')