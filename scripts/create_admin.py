"""Create an admin user interactively."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db.database import create_tables, get_session_factory
from app.models.auth import UserCreate, UserRole
from app.security.auth_manager import AuthManager


def main():
    create_tables()
    factory = get_session_factory()
    db = factory()

    try:
        email = input("Admin email: ").strip()
        password = input("Admin password: ").strip()

        if not email or not password:
            print("Email and password are required.")
            return

        if len(password) < 8:
            print("Password must be at least 8 characters.")
            return

        auth = AuthManager(db)
        existing = auth._user_repo.get_by_email(email)
        if existing:
            print(f"User '{email}' already exists.")
            return

        data = UserCreate(email=email, password=password, role=UserRole.ADMIN)
        user = auth.register_user(data)
        print(f"Admin user created: id={user.id}, email={user.email}, role={user.role}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
