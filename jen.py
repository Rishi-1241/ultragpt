import jwt

# Payload (can include more data as needed)
payload = {"username": "prakhar@gmail.com"}

# Secret key
secret_key = "h1u2m3a4n5i6z7e8"

# Generate token
token = jwt.encode(payload, secret_key, algorithm="HS256")
print(token)