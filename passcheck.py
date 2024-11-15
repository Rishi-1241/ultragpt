from werkzeug.security import check_password_hash

hashed_password_from_db = "scrypt:32768:8:1$s5CYot2rFZ25MkO8$1d3b5f108eaf6f51b7d02f2306c970c7721cb6323c95b1ab9f7e70ad25a8934d8dab4a77902dcc1295decdf1597cc00438b49a73aa62bd49e8cb1240a1ddb4d9"
user_entered_password = "abc123"

if check_password_hash(hashed_password_from_db, user_entered_password):
    print("Password matches!")
