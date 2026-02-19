"""Create admin user with simple password hashing (for demo only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
from datetime import datetime

from app.db.database import create_tables, get_session_factory
from app.db.models import User
from app.models.auth import UserRole


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

        # Check if user exists
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"User '{email}' already exists.")
            return

        # Simple SHA256 hash (for demo only - not secure for production)
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        user = User(
            email=email,
            hashed_password=hashed_password,
            role=UserRole.ADMIN.value,
            is_active=True,
            created_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        print(f"Admin user created: id={user.id}, email={user.email}, role={user.role}")
        print("You can now login with these credentials.")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
