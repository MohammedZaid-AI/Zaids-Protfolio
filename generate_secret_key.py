"""
Generate a secure SECRET_KEY for Flask application
"""

import secrets

print("=" * 60)
print("FLASK SECRET KEY GENERATOR")
print("=" * 60)
print("\nGenerated SECRET_KEY:")
print("-" * 60)
secret_key = secrets.token_hex(32)
print(secret_key)
print("-" * 60)
print("\nAdd this to your Vercel environment variables:")
print(f"SECRET_KEY={secret_key}")
print("\nFor local development, add to your .env file:")
print(f"SECRET_KEY={secret_key}")
print("=" * 60)
