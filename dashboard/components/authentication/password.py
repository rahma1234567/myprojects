import bcrypt

passwords = {
    "admin": "admin123",
    "user": "user123"}

print("Copy these hashed passwords into credentials.yaml:")
for username, pwd in passwords.items():
    hashed = bcrypt.hashpw(pwd.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    print(f"{username} -> {hashed}")