from fastapi import APIRouter, HTTPException, Depends, status
from sqlmodel import Session, select
from typing import List, Dict, Optional
from pydantic import BaseModel
import json
from ..db.base import get_session
from ..db.models import User, UserToken, Organization, Job
from ..core.token_manager import token_manager, store_user_github_token, get_token_for_user

class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    organization: Optional[str] = None
    github_username: Optional[str] = None

class GitHubTokenRequest(BaseModel):
    github_token: str
    token_name: Optional[str] = "Code Review Agent Token"

router = APIRouter(prefix="/api", tags=["users"])

@router.get("/profile")
async def get_current_user_profile(session: Session = Depends(get_session)):
    """Get current user profile, create default user if none exists"""
    try:
        # Check if user exists
        user_result = await session.execute(
            select(User).where(User.id == 1)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            # Create default user
            user = User(
                username="demo_user",
                email="demo@example.com",
                full_name="Demo User",
                organization="Demo Org",
                github_username="demo_user",
                role="user"
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
        
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "organization": user.organization,
            "github_username": user.github_username,
            "role": user.role.value if hasattr(user.role, 'value') else user.role
        }
    except Exception as e:
        # Fallback to static response if database operation fails
        return {
            "id": 1,
            "username": "demo_user",
            "email": "demo@example.com",
            "full_name": "Demo User",
            "organization": "Demo Org",
            "github_username": "demo_user",
            "role": "user"
        }

@router.post("/register")
async def register_user(
    username: str,
    email: str,
    full_name: Optional[str] = None,
    organization: Optional[str] = None,
    github_username: Optional[str] = None,
    session: Session = Depends(get_session)
):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = await session.execute(
            select(User).where(User.username == username)
        )
        if existing_user.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Check if email already exists
        existing_email = await session.execute(
            select(User).where(User.email == email)
        )
        if existing_email.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            organization=organization,
            github_username=github_username
        )
        
        session.add(user)
        await session.commit()
        await session.refresh(user)
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "organization": user.organization,
                "github_username": user.github_username,
                "role": user.role.value
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register user: {str(e)}"
        )

@router.post("/{user_id}/github-token")
async def add_github_token(
    user_id: int,
    token_data: GitHubTokenRequest,
    session: Session = Depends(get_session)
):
    """Add or update a user's GitHub token"""
    try:
        # Verify user exists
        user_result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Validate the GitHub token
        token_validation = await token_manager.validate_token(token_data.github_token)
        if not token_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid GitHub token: {token_validation.get('error', 'Unknown error')}"
            )
        
        # Store the token securely
        user_token = await store_user_github_token(user_id, token_data.github_token, token_data.token_name)
        
        return {
            "success": True,
            "message": "GitHub token added successfully",
            "token_info": {
                "id": user_token.id,
                "token_name": user_token.token_name,
                "scopes": json.loads(user_token.scopes),
                "expires_at": user_token.expires_at,
                "last_used": user_token.last_used
            },
            "github_user": token_validation["user"]["login"],
            "organizations": [org["login"] for org in token_validation["organizations"]]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add GitHub token: {str(e)}"
        )

@router.get("/{user_id}/github-tokens")
async def list_github_tokens(
    user_id: int,
    session: Session = Depends(get_session)
):
    """List all GitHub tokens for a user"""
    try:
        # Verify user exists
        user_result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get user's tokens
        tokens = await token_manager.list_user_tokens(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "tokens": tokens
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tokens: {str(e)}"
        )

@router.delete("/{user_id}/github-token/{token_id}")
async def revoke_github_token(
    user_id: int,
    token_id: int,
    session: Session = Depends(get_session)
):
    """Revoke a user's GitHub token"""
    try:
        # Verify user exists
        user_result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Revoke the token
        success = await token_manager.revoke_user_token(user_id)
        
        if success:
            return {
                "success": True,
                "message": "GitHub token revoked successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Token not found or already revoked"
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke token: {str(e)}"
        )

@router.get("/{user_id}/jobs")
async def get_user_jobs(
    user_id: int,
    session: Session = Depends(get_session)
):
    """Get all jobs created by a user"""
    try:
        # Verify user exists
        user_result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get user's jobs
        jobs_result = await session.execute(
            select(Job).where(Job.user_id == user_id).order_by(Job.created_at.desc())
        )
        jobs = jobs_result.scalars().all()
        
        return {
            "success": True,
            "user_id": user_id,
            "total_jobs": len(jobs),
            "jobs": [
                {
                    "id": job.id,
                    "repo_url": job.repo_url,
                    "status": job.status.value,
                    "created_at": job.created_at,
                    "completed_at": job.completed_at,
                    "findings_count": job.findings_count,
                    "is_pr_analysis": job.is_pr_analysis
                }
                for job in jobs
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user jobs: {str(e)}"
        )

@router.get("/{user_id}/profile")
async def get_user_profile(
    user_id: int,
    session: Session = Depends(get_session)
):
    """Get a user's profile information"""
    try:
        # Get user with token info
        user_result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get user's active tokens
        tokens = await token_manager.list_user_tokens(user_id)
        active_tokens = [t for t in tokens if t["is_active"]]
        
        # Get user's job statistics
        jobs_result = await session.execute(
            select(Job).where(Job.user_id == user_id)
        )
        jobs = jobs_result.scalars().all()
        
        completed_jobs = [j for j in jobs if j.status.value == "completed"]
        total_findings = sum(j.findings_count for j in completed_jobs)
        
        return {
            "success": True,
            "profile": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "organization": user.organization,
                "github_username": user.github_username,
                "role": user.role.value,
                "created_at": user.created_at,
                "last_login": user.last_login,
                "is_active": user.is_active
            },
            "statistics": {
                "total_jobs": len(jobs),
                "completed_jobs": len(completed_jobs),
                "total_findings": total_findings,
                "active_github_tokens": len(active_tokens)
            },
            "github_tokens": active_tokens
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user profile: {str(e)}"
        )

@router.put("/{user_id}/profile")
async def update_user_profile(
    user_id: int,
    profile_data: ProfileUpdateRequest,
    session: Session = Depends(get_session)
):
    """Update a user's profile information"""
    try:
        # Verify user exists
        user_result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update fields if provided
        if profile_data.full_name is not None:
            user.full_name = profile_data.full_name
        if profile_data.organization is not None:
            user.organization = profile_data.organization
        if profile_data.github_username is not None:
            user.github_username = profile_data.github_username
        
        await session.commit()
        await session.refresh(user)
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "profile": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "organization": user.organization,
                "github_username": user.github_username,
                "role": user.role.value
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile: {str(e)}"
        )

@router.post("/{user_id}/test-token")
async def test_github_token(
    user_id: int,
    session: Session = Depends(get_session)
):
    """Test if a user's GitHub token is still valid"""
    try:
        # Get user's active token
        token = await get_token_for_user(user_id)
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active GitHub token found"
            )
        
        # Validate the token
        validation = await token_manager.validate_token(token)
        
        return {
            "success": True,
            "token_valid": validation["valid"],
            "github_user": validation.get("user", {}).get("login") if validation["valid"] else None,
            "organizations": [org["login"] for org in validation.get("organizations", [])] if validation["valid"] else [],
            "scopes": validation.get("scopes", []),
            "error": validation.get("error") if not validation["valid"] else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test token: {str(e)}"
        )
