
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt_file(file_bytes: bytes) -> bytes:
    key = bytes.fromhex(os.environ["FILE_ENCRYPTION_KEY"])

    if len(key) != 32:
        raise ValueError("Encryption key must be 32 bytes")

    aesgcm = AESGCM(key)

    nonce = os.urandom(12)

    encrypted_data = aesgcm.encrypt(
        nonce,
        file_bytes,
        None
    )

    return nonce + encrypted_data


def decrypt_file(encrypted_bytes: bytes) -> bytes:
    key = bytes.fromhex(os.environ["FILE_ENCRYPTION_KEY"])

    if len(key) != 32:
        raise ValueError("Encryption key must be 32 bytes")

    nonce = encrypted_bytes[:12]
    ciphertext = encrypted_bytes[12:]

    aesgcm = AESGCM(key)

    return aesgcm.decrypt(
        nonce,
        ciphertext,
        None
    )
