from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession 
from app.database import get_session 
from app.models import User 
from sqlalchemy.future import select 

router = APIRouter()

# GET method to return all users 
@router.get('/', tags = ['Users'])
async def get_all_users(limit: int = 1000, offset: int = 0, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(User).offset(offset).limit(limit)
    )
    return result.scalars().all()

# DELETE method to delete a user by user id 
@router.delete('/{user_id}', tags=['Users'])
async def delete_user(user_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    # if the user does not exist by that id, raise error 
    if not user:
        raise HTTPException(staus_code = 404, detail = 'User not found')
    
    await session.delete(user)
    await session.commit()
    return {"message": f"User {user_id} deleted successfully"}

# PUT method to update a user's attributes 
@router.put('/{user_id}', tags=['Users'])
async def update_user(user_id: int, updates: dict, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code = 404, detail = 'User not found')
    
    for key, value in updates.items(): 
        if hasattr(user, key):
            setattr(user, key, value)
    
    await session.commit()
    return {"message": f"User {user_id} updated successfully"}
    