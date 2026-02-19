from sqlalchemy.orm import Session

from app.db.models import User
from app.models.auth import UserCreate, UserRole


class UserRepository:
    def __init__(self, db: Session):
        self._db = db

    def create(self, user_data: UserCreate, hashed_password: str) -> User:
        user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            role=user_data.role,
        )
        self._db.add(user)
        self._db.commit()
        self._db.refresh(user)
        return user

    def get_by_id(self, user_id: int) -> User | None:
        return self._db.query(User).filter(User.id == user_id).first()

    def get_by_email(self, email: str) -> User | None:
        return self._db.query(User).filter(User.email == email).first()

    def list_all(self, offset: int = 0, limit: int = 20) -> list[User]:
        return self._db.query(User).offset(offset).limit(limit).all()

    def count(self) -> int:
        return self._db.query(User).count()

    def update_role(self, user_id: int, role: UserRole) -> User | None:
        user = self.get_by_id(user_id)
        if user:
            user.role = role
            self._db.commit()
            self._db.refresh(user)
        return user

    def delete(self, user_id: int) -> bool:
        user = self.get_by_id(user_id)
        if user:
            self._db.delete(user)
            self._db.commit()
            return True
        return False
