import os
import json
import hashlib
import hmac
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from ..db.models import User, UserToken, Organization
from ..core.settings import settings

class TokenManager:
    """Secure management of GitHub tokens for multiple users"""
    
    def __init__(self):
        # Get encryption key from environment or generate one
        self.encryption_key = os.getenv("ENCRYPTION_KEY")
        if not self.encryption_key:
            # Generate a new key if none exists
            self.encryption_key = Fernet.generate_key().decode()
            print(f"⚠️  Generated new encryption key. Add this to your .env file:")
            print(f"ENCRYPTION_KEY={self.encryption_key}")
        
        self.cipher = Fernet(self.encryption_key.encode())
    
    def encrypt_token(self, token: str) -> str:
        """Encrypt a GitHub token for secure storage"""
        return self.cipher.encrypt(token.encode()).decode()
    
    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt a stored GitHub token"""
        return self.cipher.decrypt(encrypted_token.encode()).decode()
    
    def hash_token(self, token: str) -> str:
        """Create a hash of the token for identification"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    async def store_user_token(self, user_id: int, token: str, token_name: str = "Code Review Agent Token") -> UserToken:
        """Store a user's GitHub token securely"""
        from ..db.base import SessionLocal
        
        async with SessionLocal() as session:
            # Check if user already has a token
            existing_token = await session.execute(
                "SELECT * FROM usertoken WHERE user_id = :user_id AND is_active = 1"
            )
            
            if existing_token:
                # Update existing token
                existing_token.token_hash = self.encrypt_token(token)
                existing_token.token_name = token_name
                existing_token.last_used = datetime.utcnow()
                await session.commit()
                return existing_token
            else:
                # Create new token
                user_token = UserToken(
                    user_id=user_id,
                    token_hash=self.encrypt_token(token),
                    token_name=token_name,
                    scopes=json.dumps(self._get_token_scopes(token)),
                    expires_at=self._get_token_expiry(token),
                    last_used=datetime.utcnow()
                )
                session.add(user_token)
                await session.commit()
                await session.refresh(user_token)
                return user_token
    
    async def get_user_token(self, user_id: int) -> Optional[str]:
        """Get a user's decrypted GitHub token"""
        from ..db.base import SessionLocal
        
        async with SessionLocal() as session:
            result = await session.execute(
                "SELECT * FROM usertoken WHERE user_id = :user_id AND is_active = 1"
            )
            user_token = result.scalar_one_or_none()
            
            if user_token and user_token.token_hash:
                return self.decrypt_token(user_token.token_hash)
            return None
    
    async def get_org_token(self, org_id: int) -> Optional[str]:
        """Get an organization's decrypted GitHub token"""
        from ..db.base import SessionLocal
        
        async with SessionLocal() as session:
            result = await session.execute(
                "SELECT * FROM organization WHERE id = :org_id"
            )
            org = result.scalar_one_or_none()
            
            if org and org.org_token_hash:
                return self.decrypt_token(org.org_token_hash)
            return None
    
    async def get_best_token_for_repo(self, repo_url: str, user_id: Optional[int] = None, org_id: Optional[int] = None) -> Optional[str]:
        """Get the best available token for a repository"""
        # Priority order: User token > Org token > Global token
        
        # 1. Try user token first
        if user_id:
            user_token = await self.get_user_token(user_id)
            if user_token and await self._can_access_repo(repo_url, user_token):
                return user_token
        
        # 2. Try organization token
        if org_id:
            org_token = await self.get_org_token(org_id)
            if org_token and await self._can_access_repo(repo_url, org_token):
                return org_token
        
        # 3. Fall back to global token from settings
        global_token = settings.GITHUB_TOKEN
        if global_token and await self._can_access_repo(repo_url, global_token):
            return global_token
        
        return None
    
    async def _can_access_repo(self, repo_url: str, token: str) -> bool:
        """Check if a token can access a specific repository"""
        try:
            import aiohttp
            
            # Parse repo URL to get owner and repo name
            parts = repo_url.rstrip('/').split('/')
            if len(parts) < 2:
                return False
            
            owner = parts[-2]
            repo = parts[-1]
            
            # Test API access
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            url = f"https://api.github.com/repos/{owner}/{repo}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    return response.status == 200
                    
        except Exception:
            return False
    
    def _get_token_scopes(self, token: str) -> List[str]:
        """Get the scopes/permissions of a GitHub token"""
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # This would require a synchronous call, so we'll return basic scopes
            # In a real implementation, you'd make an async call to get scopes
            return ["repo", "repo:status", "repo:public_repo"]
            
        except Exception:
            return ["repo", "repo:status", "repo:public_repo"]
    
    def _get_token_expiry(self, token: str) -> Optional[datetime]:
        """Get the expiration date of a GitHub token"""
        # GitHub tokens can have expiration dates
        # For now, we'll set a default 90-day expiration
        return datetime.utcnow() + timedelta(days=90)
    
    async def revoke_user_token(self, user_id: int) -> bool:
        """Revoke a user's GitHub token"""
        from ..db.base import SessionLocal
        
        async with SessionLocal() as session:
            result = await session.execute(
                "UPDATE usertoken SET is_active = 0 WHERE user_id = :user_id"
            )
            await session.commit()
            return result.rowcount > 0
    
    async def list_user_tokens(self, user_id: int) -> List[Dict]:
        """List all tokens for a user"""
        from ..db.base import SessionLocal
        
        async with SessionLocal() as session:
            result = await session.execute(
                "SELECT * FROM usertoken WHERE user_id = :user_id ORDER BY created_at DESC"
            )
            tokens = result.scalars().all()
            
            return [
                {
                    "id": token.id,
                    "token_name": token.token_name,
                    "scopes": json.loads(token.scopes),
                    "expires_at": token.expires_at,
                    "last_used": token.last_used,
                    "is_active": token.is_active,
                    "created_at": token.created_at
                }
                for token in tokens
            ]
    
    async def validate_token(self, token: str) -> Dict:
        """Validate a GitHub token and return user info"""
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            async with aiohttp.ClientSession() as session:
                # Get user info
                async with session.get("https://api.github.com/user", headers=headers) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        
                        # Get user's organizations
                        async with session.get("https://api.github.com/user/orgs", headers=headers) as org_response:
                            orgs = []
                            if org_response.status == 200:
                                orgs = await org_response.json()
                        
                        return {
                            "valid": True,
                            "user": user_data,
                            "organizations": orgs,
                            "scopes": self._get_token_scopes(token)
                        }
                    else:
                        return {
                            "valid": False,
                            "error": f"GitHub API returned status {response.status}"
                        }
                        
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }

# Global token manager instance
token_manager = TokenManager()

# Convenience functions
async def get_token_for_user(user_id: int) -> Optional[str]:
    """Get GitHub token for a specific user"""
    return await token_manager.get_user_token(user_id)

async def get_token_for_repo(repo_url: str, user_id: Optional[int] = None, org_id: Optional[int] = None) -> Optional[str]:
    """Get the best available token for a repository"""
    return await token_manager.get_best_token_for_repo(repo_url, user_id, org_id)

async def store_user_github_token(user_id: int, token: str, token_name: str = "Code Review Agent Token") -> UserToken:
    """Store a user's GitHub token securely"""
    return await token_manager.store_user_token(user_id, token, token_name)
