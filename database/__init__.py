import sqlite3
import hashlib
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GestureDatabase:
    def __init__(self, db_path: str = 'gesture_data.db'):
        """Initialize database connection and create tables"""
        self.db_path = db_path
        self.conn = None
        self.create_tables()
        self._init_admin_user()

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            cursor.execute('SELECT 1')
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
        finally:
            self._close_connection()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with foreign key support"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA foreign_keys = ON')
        return conn

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_tables(self) -> None:
        """Create necessary database tables if they don't exist"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            
            # Create gestures table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gestures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create gesture_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gesture_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_id INTEGER NOT NULL,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (gesture_id) REFERENCES gestures(id) ON DELETE CASCADE
                )
            ''')

            # Create users table with better password handling
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    is_admin BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    login_attempts INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            self._close_connection()

    def _init_admin_user(self) -> None:
        """Initialize default admin users if not exist"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            
            # Default admin credentials
            default_admins = [
                ('admin', 'admin123', True),
            ]
            
            for username, password, is_admin in default_admins:
                cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
                if cursor.fetchone()[0] == 0:
                    hashed_password = self._hash_password(password)
                    cursor.execute('''
                        INSERT INTO users (username, password_hash, is_admin, is_active)
                        VALUES (?, ?, ?, TRUE)
                    ''', (username, hashed_password, is_admin))
                    logger.info(f"Default admin user '{username}' created")
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error initializing admin users: {e}")
            if self.conn:
                self.conn.rollback()
        finally:
            self._close_connection()

    def authenticate_user(self, username: str, password: str) -> Optional[Tuple[int, str, bool]]:
        """Authenticate user and update login info"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            
            hashed_password = self._hash_password(password)
            cursor.execute('''
                SELECT id, username, is_admin 
                FROM users 
                WHERE username = ? AND password_hash = ? AND is_active = TRUE
            ''', (username, hashed_password))
            
            user = cursor.fetchone()
            if user:
                # Reset login attempts and update last login
                cursor.execute('''
                    UPDATE users 
                    SET login_attempts = 0, last_login = CURRENT_TIMESTAMP
                    WHERE username = ?
                ''', (username,))
                logger.info(f"User '{username}' authenticated successfully")
            else:
                # Increment login attempts
                cursor.execute('''
                    UPDATE users 
                    SET login_attempts = login_attempts + 1
                    WHERE username = ?
                ''', (username,))
                logger.warning(f"Failed login attempt for user '{username}'")
            
            self.conn.commit()
            return user
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            if self.conn:
                self.conn.rollback()
            return None
        finally:
            self._close_connection()

    def verify_user(self, username: str, password: str) -> Optional[dict]:
        """Verify user credentials and return user info"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            
            hashed_password = self._hash_password(password)
            cursor.execute('''
                SELECT id, username, is_admin 
                FROM users 
                WHERE username = ? 
                AND password_hash = ? 
                AND is_active = TRUE
            ''', (username, hashed_password))
            
            user = cursor.fetchone()
            if user:
                return {
                    'id': user[0],
                    'username': user[1],  # Make sure username is returned
                    'is_admin': bool(user[2])
                }
            return None
        except Exception as e:
            logger.error(f"Error verifying user: {e}")
            return None
        finally:
            self._close_connection()

    def initialize_default_gestures(self) -> bool:
        """Initialize default gestures - called only from admin panel"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM gestures')
            if cursor.fetchone()[0] == 0:
                default_gestures = [
                    ('Hello', 'Tangan terbuka'),
                    ('Peace', 'Jari Telunjuk dan tengah terangkat'),
                    ('Iloveyou', 'Jari kelingking, telunjuk, dan jempol terangkat'),
                    ('Yes', 'Tangan Mengepal kedepan'),
                    ('No', 'Jari jempol, telunjuk, dan tengah kedepan')
                ]
                
                cursor.executemany('''
                    INSERT OR IGNORE INTO gestures (name, description)
                    VALUES (?, ?)
                ''', default_gestures)
                
                self.conn.commit()
                logger.info("Default gestures initialized")
                return True
            return False
        except Exception as e:
            logger.error(f"Error initializing default gestures: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self._close_connection()

    def get_all_gestures(self) -> List[Tuple[int, str, str]]:
        """Get all gestures from database"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            cursor.execute('SELECT id, name, description FROM gestures ORDER BY id')
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting gestures: {e}")
            return []
        finally:
            self._close_connection()
    


    def add_gesture(self, name: str, description: str) -> bool:
        """Add a new gesture to database"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO gestures (name, description)
                VALUES (?, ?)
            ''', (name, description))
            self.conn.commit()
            logger.info(f"Gesture '{name}' added successfully")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Gesture '{name}' already exists")
            return False
        except Exception as e:
            logger.error(f"Error adding gesture: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self._close_connection()

    def update_gesture(self, id: int, name: str, description: str) -> bool:
        """Update existing gesture details"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE gestures 
                SET name = ?, description = ?
                WHERE id = ?
            ''', (name, description, id))
            success = cursor.rowcount > 0
            self.conn.commit()
            if success:
                logger.info(f"Gesture {id} updated successfully")
            return success
        except sqlite3.IntegrityError:
            logger.warning(f"Gesture name '{name}' already exists")
            return False
        except Exception as e:
            logger.error(f"Error updating gesture: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self._close_connection()

    def delete_gesture(self, id: int) -> bool:
        """Delete gesture and its history"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM gestures WHERE id = ?', (id,))
            success = cursor.rowcount > 0
            self.conn.commit()
            if success:
                logger.info(f"Gesture {id} deleted successfully")
            return success
        except Exception as e:
            logger.error(f"Error deleting gesture: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self._close_connection()

    def log_gesture(self, gesture_name: str) -> bool:
        """Log detected gesture to history"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            
            cursor.execute('SELECT id FROM gestures WHERE name = ?', (gesture_name,))
            gesture_id = cursor.fetchone()
            
            if gesture_id:
                cursor.execute('''
                    INSERT INTO gesture_history (gesture_id, detected_at)
                    VALUES (?, ?)
                ''', (gesture_id[0], datetime.now()))
                self.conn.commit()
                logger.debug(f"Gesture '{gesture_name}' logged")
                return True
            return False
        except Exception as e:
            logger.error(f"Error logging gesture: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self._close_connection()

    def get_gesture_stats(self) -> List[Tuple[str, str, int]]:
        """Get statistics of gesture detections"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT g.name, g.description, COUNT(h.id) as count
                FROM gestures g
                LEFT JOIN gesture_history h ON g.id = h.gesture_id
                GROUP BY g.name, g.description
                ORDER BY count DESC, g.name
            ''')
            
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting gesture stats: {e}")
            return []
        finally:
            self._close_connection()

    def get_gesture_count(self, gesture_name: str) -> int:
        """Get the count of a specific gesture"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(h.id) 
                FROM gesture_history h
                JOIN gestures g ON h.gesture_id = g.id
                WHERE g.name = ?
            ''', (gesture_name,))
            
            count = cursor.fetchone()[0]
            return count if count else 0
        except Exception as e:
            logger.error(f"Error getting gesture count: {e}")
            return 0
        finally:
            self._close_connection()

    def get_all_users(self) -> List[Tuple[int, str, bool]]:
        """Get all users from database"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id, username, is_admin 
                FROM users 
                WHERE is_active = TRUE 
                ORDER BY id
            ''')
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return []
        finally:
            self._close_connection()

    def add_user(self, username: str, password: str, is_admin: bool = False) -> bool:
        """Add a new user with hashed password"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            
            hashed_password = self._hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, password_hash, is_admin, is_active)
                VALUES (?, ?, ?, TRUE)
            ''', (username, hashed_password, is_admin))
            
            self.conn.commit()
            logger.info(f"User '{username}' added successfully")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"User '{username}' already exists")
            return False
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self._close_connection()

    def update_user(self, id: int, username: str, 
                   password: Optional[str] = None, 
                   is_admin: Optional[bool] = None) -> bool:
        """Update user details with optional password change"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            
            if password:
                hashed_password = self._hash_password(password)
                cursor.execute('''
                    UPDATE users 
                    SET username = ?, password_hash = ?, is_admin = ?
                    WHERE id = ? AND username != 'admin'
                ''', (username, hashed_password, is_admin, id))
            else:
                cursor.execute('''
                    UPDATE users 
                    SET username = ?, is_admin = ?
                    WHERE id = ? AND username != 'admin'
                ''', (username, is_admin, id))
            
            success = cursor.rowcount > 0
            self.conn.commit()
            if success:
                logger.info(f"User {id} updated successfully")
            return success
        except sqlite3.IntegrityError:
            logger.warning(f"Username '{username}' already exists")
            return False
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self._close_connection()

    def delete_user(self, id: int) -> bool:
        """Soft delete user by setting is_active to FALSE"""
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET is_active = FALSE 
                WHERE id = ? AND username != 'admin'
            ''', (id,))
            success = cursor.rowcount > 0
            self.conn.commit()
            if success:
                logger.info(f"User {id} deactivated successfully")
            return success
        except Exception as e:
            logger.error(f"Error deactivating user: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self._close_connection()

    def _close_connection(self) -> None:
        """Helper method to close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def close(self) -> None:
        """Safely close database connection"""
        try:
            self._close_connection()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")

# Add at the bottom of the file
if __name__ == '__main__':
    try:
        print("Initializing database...")
        db = GestureDatabase()
        
        # Add default admin users
        print("\nAdding admin users...")
        admin_users = [
            ('admin', 'admin123', True),
            ('superadmin', 'super123', True),
            ('dosen', 'dosen123', True)
        ]
        
        for username, password, is_admin in admin_users:
            if db.add_user(username, password, is_admin):
                print(f"Added user: {username}")
            else:
                print(f"Failed to add user: {username}")
        
        # Initialize default gestures
        print("\nInitializing default gestures...")
        if db.initialize_default_gestures():
            print("Default gestures added successfully")
        
        print("\nDatabase initialization completed!")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
    finally:
        if 'db' in locals():
            db.close()