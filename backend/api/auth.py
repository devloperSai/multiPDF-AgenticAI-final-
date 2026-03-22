import os
import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession
from pydantic import BaseModel, EmailStr, field_validator
import bcrypt
import jwt

from models.database import get_db
from models.schema import User

router = APIRouter(prefix="/auth", tags=["auth"])

JWT_SECRET    = os.getenv("JWT_SECRET", "changeme-use-a-long-random-string-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 72   # token valid for 3 days


# ── helpers ───────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_token(user_id: str, email: str, name: str) -> str:
    payload = {
        "sub":   user_id,
        "email": email,
        "name":  name,
        "exp":   datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


# ── dependency — use in any protected endpoint ────────────────────────────────

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

bearer_scheme = HTTPBearer(auto_error=False)

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: DBSession = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    payload = decode_token(credentials.credentials)
    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


# ── request / response schemas ────────────────────────────────────────────────

class SignupRequest(BaseModel):
    name:     str
    email:    EmailStr
    password: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v):
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Full name must be at least 2 characters.")
        if not any(c.isalpha() for c in v):
            raise ValueError("Name must contain letters.")
        return v

    @field_validator("password")
    @classmethod
    def password_strong_enough(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters.")
        return v


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    user:  dict


# ── routes ────────────────────────────────────────────────────────────────────

@router.post("/signup", response_model=AuthResponse, status_code=201)
def signup(body: SignupRequest, db: DBSession = Depends(get_db)):
    # Check duplicate email
    existing = db.query(User).filter(User.email == body.email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    user = User(
        name          = body.name.strip(),
        email         = body.email.lower(),
        password_hash = hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_token(str(user.id), user.email, user.name)
    return {
        "token": token,
        "user":  {"id": str(user.id), "name": user.name, "email": user.email},
    }


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest, db: DBSession = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email.lower()).first()

    # Same error for missing user AND wrong password — prevents email enumeration
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    token = create_token(str(user.id), user.email, user.name)
    return {
        "token": token,
        "user":  {"id": str(user.id), "name": user.name, "email": user.email},
    }


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    """Returns the currently authenticated user. Use to validate token on app load."""
    return {"id": str(current_user.id), "name": current_user.name, "email": current_user.email}