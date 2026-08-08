"""Adversarial tests for Clerk session token verification.

These exist because the previous implementation base64-decoded the token
payload and trusted it, which allowed anyone to mint an admin identity. Every
test below is an attack that used to succeed.
"""
import base64
import hashlib
import hmac
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.security.clerk_jwt import ClerkTokenVerifier, TokenVerificationError

ISSUER = "https://example.clerk.accounts.dev"
FRONTEND = "https://chat.maxmayes.io"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private_pem, public_pem


@pytest.fixture(scope="module")
def keys():
    return _keypair()


@pytest.fixture(scope="module")
def attacker_keys():
    return _keypair()


@pytest.fixture
def verifier(keys):
    _, public_pem = keys
    return ClerkTokenVerifier(
        public_key_pem=public_pem,
        issuer=ISSUER,
        authorized_parties=[FRONTEND],
    )


def _token(private_pem: str, **overrides) -> str:
    now = int(time.time())
    payload = {
        "sub": "user_legit",
        "iss": ISSUER,
        "azp": FRONTEND,
        "sid": "sess_abc",
        "iat": now,
        "nbf": now,
        "exp": now + 60,
    }
    payload.update(overrides)
    return jwt.encode(payload, private_pem, algorithm="RS256")


# --- the happy path, so the negatives below mean something -------------------


def test_valid_token_is_accepted(verifier, keys):
    private_pem, _ = keys
    result = verifier.verify(_token(private_pem))
    assert result.subject == "user_legit"
    assert result.session_id == "sess_abc"
    assert result.authorized_party == FRONTEND


# --- the attacks -------------------------------------------------------------


def test_unsigned_token_with_forged_payload_is_rejected(verifier):
    """The original exploit: base64 a payload, append junk, get admin."""
    forged = jwt.encode(
        {
            "sub": "user_attacker",
            "iss": ISSUER,
            "azp": FRONTEND,
            "exp": int(time.time()) + 3600,
            "publicMetadata": {"role": "admin"},
        },
        key="",
        algorithm="none",
    )
    with pytest.raises(TokenVerificationError):
        verifier.verify(forged)


def test_token_signed_with_attacker_key_is_rejected(verifier, attacker_keys):
    attacker_private, _ = attacker_keys
    with pytest.raises(TokenVerificationError):
        verifier.verify(_token(attacker_private, sub="user_attacker"))


def test_hmac_signed_token_using_public_key_as_secret_is_rejected(verifier, keys):
    """Algorithm-confusion: sign HS256 using the published RSA public key.

    Hand-rolled because PyJWT refuses to *create* this token — which is a good
    default, but means we have to build the attack the way an attacker would.
    """
    _, public_pem = keys
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(
        json.dumps(
            {
                "sub": "user_attacker",
                "iss": ISSUER,
                "azp": FRONTEND,
                "exp": int(time.time()) + 3600,
            }
        ).encode()
    )
    signing_input = f"{header}.{payload}".encode()
    signature = _b64url(
        hmac.new(public_pem.encode(), signing_input, hashlib.sha256).digest()
    )
    with pytest.raises(TokenVerificationError):
        verifier.verify(f"{header}.{payload}.{signature}")


def test_expired_token_is_rejected(verifier, keys):
    private_pem, _ = keys
    past = int(time.time()) - 3600
    with pytest.raises(TokenVerificationError):
        verifier.verify(_token(private_pem, iat=past, nbf=past, exp=past + 60))


def test_token_from_another_issuer_is_rejected(verifier, keys):
    private_pem, _ = keys
    with pytest.raises(TokenVerificationError):
        verifier.verify(_token(private_pem, iss="https://evil.clerk.accounts.dev"))


def test_token_for_unauthorized_frontend_is_rejected(verifier, keys):
    private_pem, _ = keys
    with pytest.raises(TokenVerificationError):
        verifier.verify(_token(private_pem, azp="https://evil.example.com"))


def test_token_without_subject_is_rejected(verifier, keys):
    private_pem, _ = keys
    now = int(time.time())
    token = jwt.encode(
        {"iss": ISSUER, "azp": FRONTEND, "iat": now, "exp": now + 60},
        private_pem,
        algorithm="RS256",
    )
    with pytest.raises(TokenVerificationError):
        verifier.verify(token)


def test_token_without_expiry_is_rejected(verifier, keys):
    private_pem, _ = keys
    token = jwt.encode(
        {"sub": "user_legit", "iss": ISSUER, "azp": FRONTEND},
        private_pem,
        algorithm="RS256",
    )
    with pytest.raises(TokenVerificationError):
        verifier.verify(token)


@pytest.mark.parametrize(
    "garbage",
    ["", "   ", "not-a-jwt", "a.b", "a.b.c", "Bearer something", "..."],
)
def test_malformed_tokens_are_rejected(verifier, garbage):
    with pytest.raises(TokenVerificationError):
        verifier.verify(garbage)


def test_publicmetadata_role_in_token_is_not_trusted(verifier, keys):
    """A validly-signed token may still claim to be an admin.

    Verification must surface the claim without conferring privilege — the role
    decision belongs to the database, not the token.
    """
    private_pem, _ = keys
    result = verifier.verify(
        _token(private_pem, publicMetadata={"role": "admin"})
    )
    assert result.claims["publicMetadata"] == {"role": "admin"}
    # Nothing on VerifiedToken exposes a role or any privilege signal.
    assert not hasattr(result, "role")
    assert not hasattr(result, "is_admin")


def test_verifier_requires_a_key_source():
    with pytest.raises(ValueError):
        ClerkTokenVerifier()
