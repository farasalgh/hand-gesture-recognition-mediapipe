import os
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import GestureDatabase

def test_database():
    db = GestureDatabase()
    try:
        # Test admin users
        print("\nTesting admin users:")
        users = db.get_all_users()
        for user in users:
            print(f"User ID: {user[0]}, Username: {user[1]}, Is Admin: {user[2]}")
        
        # Test gestures
        print("\nTesting gestures:")
        gestures = db.get_all_gestures()
        for gesture in gestures:
            print(f"Gesture ID: {gesture[0]}, Name: {gesture[1]}, Description: {gesture[2]}")
            
    finally:
        db.close()

if __name__ == '__main__':
    test_database()