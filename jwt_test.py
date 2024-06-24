"""Sample JWT authentication module for testing purposes.

NOT SUITABLE FOR PRODUCTION USE.
"""
import logging
from typing import Union

import jwt
from flask import Response, make_response, request
from werkzeug.datastructures import Authorization
from mlflow.server.auth import store as auth_store

BEARER_PREFIX = "bearer "

_logger = logging.getLogger(__name__)



_auth_store = auth_store


def update_user(user_info: dict = None):
    if _auth_store.has_user(user_info["username"]) is False:
        _auth_store.create_user(user_info["username"], user_info["username"], user_info["is_admin"])
    else:
        _auth_store.update_user(user_info["username"], user_info["username"], user_info["is_admin"])



def authenticate_request() -> Union[Authorization, Response]:
    _logger.debug("Getting token")
    error_response = make_response()
    error_response.status_code = 401
    error_response.set_data(
        "You are not authenticated. Please provide a valid JWT Bearer token with the request."
    )
    error_response.headers["WWW-Authenticate"] = 'Bearer error="invalid_token"'

    token = request.headers.get("Authorization")
    token_query = request.args.get('jwt_token')
    print(token)
    print(token_query)

    if not token and token_query:
        token = token_query
    
    # token.lower().startswith(BEARER_PREFIX)

    if token is not None:
        if token.lower().startswith(BEARER_PREFIX):
            token = token[len(BEARER_PREFIX) :]  # Remove prefix
        try:
            # NOTE:
            # - This is a sample implementation for testing purposes only.
            # - Here we're using a hardcoded key, which is not secure.
            # - We also aren't validating that the user exists.
            user_info = dict()
            token_info = jwt.decode(token, "secret", algorithms=["HS256"])
            if not token_info:  # pragma: no cover
                _logger.warning("No token_info returned")
                return error_response
            user_info["username"] = token_info['username']
            user_info["is_admin"] = True
            update_user(user_info)

            return Authorization(auth_type="jwt", data=user_info)
        except jwt.exceptions.InvalidTokenError:
            pass

    _logger.warning("Missing or invalid authorization token")
    return error_response
