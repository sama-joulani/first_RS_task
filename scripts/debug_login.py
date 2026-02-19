"""Debug login to see what's happening."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib

from app.db.database import get_session_factory
from app.db.models import User
from app.security.auth_manager import AuthManager


def main():
    factory = get_session_factory()
    db = factory()

    try:
        email = input("User email: ").strip()
        password = input("Password: ").strip()

        # Find user
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"❌ User '{email}' not found in database.")
            return

        print(f"✓ User found: {user.email}")
        print(f"  Stored hash: {user.hashed_password}")
        print(f"  Hash length: {len(user.hashed_password)}")
        
        # Check if SHA256
        is_sha256 = len(user.hashed_password) == 64 and all(c in '0123456789abcdef' for c in user.hashed_password)
        print(f"  Is SHA256: {is_sha256}")
        
        # Calculate SHA256
        calculated_hash = hashlib.sha256(password.encode()).hexdigest()
        print(f"  Calculated:  {calculated_hash}")
        print(f"  Match: {calculated_hash == user.hashed_password}")
        
        # Try AuthManager
        auth = AuthManager(db)
        result = auth.verify_password(password, user.hashed_password)
        print(f"  AuthManager verify: {result}")
        
        # Try authenticate
        auth_user = auth.authenticate_user(email, password)
        if auth_user:
            print(f"✓ Authentication successful!")
        else:
            print(f"❌ Authentication failed!")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
