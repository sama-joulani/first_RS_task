"""Reset user password (for demo only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib

from app.db.database import get_session_factory
from app.db.models import User


def main():
    factory = get_session_factory()
    db = factory()

    try:
        email = input("User email: ").strip()
        new_password = input("New password: ").strip()

        if not email or not new_password:
            print("Email and password are required.")
            return

        if len(new_password) < 8:
            print("Password must be at least 8 characters.")
            return

        # Find user
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"User '{email}' not found.")
            return

        # Reset with SHA256 hash
        user.hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
        db.commit()
        
        print(f"Password reset for: {user.email}")
        print("You can now login with the new password.")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
