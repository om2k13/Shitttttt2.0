# Well-structured, secure Python code example
import hashlib
import secrets
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class UserCredentials:
    username: str
    password_hash: str
    salt: str

class SecureUserManager:
    def __init__(self):
        self.users: Dict[str, UserCredentials] = {}
    
    def create_user(self, username: str, password: str) -> bool:
        """Create a new user with secure password hashing"""
        if self.user_exists(username):
            return False
        
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(password, salt)
        
        self.users[username] = UserCredentials(
            username=username,
            password_hash=password_hash,
            salt=salt
        )
        
        logger.info(f"User created: {username}")
        return True
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user with secure password verification"""
        if not self.user_exists(username):
            return False
        
        user = self.users[username]
        expected_hash = self._hash_password(password, user.salt)
        
        return secrets.compare_digest(expected_hash, user.password_hash)
    
    def user_exists(self, username: str) -> bool:
        """Check if user exists"""
        return username in self.users
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using secure algorithm"""
        return hashlib.pbkdf2_hmac('sha256', 
                                   password.encode('utf-8'), 
                                   salt.encode('utf-8'), 
                                   100000).hex()

def validate_input(user_input: str) -> Optional[str]:
    """Validate and sanitize user input"""
    if not user_input or len(user_input) > 1000:
        return None
    
    # Remove potentially dangerous characters
    sanitized = ''.join(c for c in user_input if c.isalnum() or c in ' .-_')
    return sanitized.strip()

def process_data_safely(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process data with proper error handling"""
    try:
        result = {
            'processed': True,
            'timestamp': secrets.token_hex(16),
            'data_length': len(str(data))
        }
        return result
    
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return {'processed': False, 'error': 'Processing failed'}
