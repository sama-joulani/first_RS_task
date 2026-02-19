"""Test JWT token generation and verification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.database import get_session_factory
from app.db.models import User
from app.security.auth_manager import AuthManager
from app.models.auth import UserRole


def main():
    factory = get_session_factory()
    db = factory()

    try:
        email = input("User email: ").strip()
        
        # Find user
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"❌ User '{email}' not found.")
            return
        
        print(f"✓ User found: {user.email}")
        print(f"  ID: {user.id}")
        print(f"  Role: {user.role}")
        print(f"  Is active: {user.is_active}")
        
        # Create token
        auth = AuthManager(db)
        token = auth.create_access_token(user.id, UserRole(user.role))
        print(f"\n✓ Token created: {token[:50]}...")
        
        # Verify token
        payload = auth.verify_token(token)
        if payload:
            print(f"✓ Token verified!")
            print(f"  User ID: {payload.sub}")
            print(f"  Role: {payload.role}")
        else:
            print(f"❌ Token verification failed!")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
