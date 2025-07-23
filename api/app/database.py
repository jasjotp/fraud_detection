"""
Async SQLAlchemy 2.0 engine / session factory for the fraud-detection service.


Usage inside a FastAPI router:

    from fastapi import APIRouter, Depends
    from sqlalchemy.ext.asyncio import AsyncSession
    from .database import get_session
    from . import models as m

    router = APIRouter()

    @router.post("/users")
    async def create_user(user: m.UserCreate, db: AsyncSession = Depends(get_session)):
        db.add(m.User(**user.model_dump()))
        await db.commit()
        await db.refresh(user)
        return user
"""
from __future__ import annotations 
import os 
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models import Base 

# configure runtime configurationL URL comes from .env file 
DB_URL: str = os.environ.get('DB_URL', 'postgresql+asyncpg://airflow:airflow@postgres/airflow')
ENGINE_ECHO: bool = os.getenv('SQL_ECHO', '0') == '1'

engine: AsyncEngine = create_async_engine(
    DB_URL, 
    echo = ENGINE_ECHO, 
    pool_pre_ping = True,
    future = True,
)

from sqlalchemy.ext.asyncio import AsyncSession

# session factory and FastAPI dependecny 
async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit = False
)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    # FastAPI dependecy that yields a DB session *and always closes it*
    async with async_session_factory() as session:
        try:
            yield session 
        finally:
            await session.close()

# helper function to create all the tables 
async def create_all_tables() -> None:
    # create all tables from models.metadata
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print('All tables created')

# main function: python -m app.database create
if __name__ == "__main__":
    import asyncio
    import sys 

    if len(sys.argv) == 2 and sys.argv[1] == 'create':
        asyncio.run(create_all_tables())
    else:
        print('Usage: python -m app.database create')