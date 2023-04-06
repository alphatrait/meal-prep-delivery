from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()

# Authentication
@auth.verify_password
def verify_password(username, password):
    # Check if the provided username and password are correct
    if username == 'username' and password == 'password':
        return True
    return False
