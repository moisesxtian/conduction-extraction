import httpx
import time
from typing import Union
from pydantic import BaseModel


class LarkBaseHTTPException(Exception):
    def __init__(self, code: int, msg: str) -> None:
        self.code = code
        self.msg = msg

        super().__init__(f"LarkBaseHTTPError {code}: {msg}")
        
# Authentication, Access tokens

GET_TENANT_ACCESS_TOKEN_INTERNAL_URL: str = "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal"
GET_APP_ACCESS_TOKEN_INTERNAL_URL: str = "https://open.larksuite.com/open-apis/auth/v3/app_access_token/internal"
GET_USER_ACCESS_TOKEN_URL: str = "https://open.larksuite.com/open-apis/authen/v1/oidc/access_token"
GET_USER_INFORMATION_URL: str = "https://open.larksuite.com/open-apis/authen/v1/user_info" 



class LarkBaseTokenResponse(BaseModel):
    code: int
    msg: str
    expire: int


class AppTokenResponse(LarkBaseTokenResponse):
    app_access_token: str


class TenantTokenResponse(LarkBaseTokenResponse):
    tenant_access_token: str


class UserTokenDataResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    refresh_expires_in: int
    scope: str


class UserTokenResponse(BaseModel):
    code: int
    message: str
    data: UserTokenDataResponse


class UserInformationDataResponse(BaseModel):
    avatar_thumb: str
    avatar_url: str
    en_name: str
    name: str
    tenant_key: str
    union_id: str
    user_id: str
    

class UserInformationResponse(BaseModel):
    code: int
    data: Union[UserInformationDataResponse, None]
    msg: str


class TokenManager:
    def __init__(self, app_id: str, app_secret: str) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self._cache = {}

    async def get_user_access_token(
        self,
        code: str,
        grant_type: str = "authorization_code"
    ):
        app_token_response = await self.get_app_access_token()

        headers = {
            "Authorization": f"Bearer {app_token_response.app_access_token}"
        }

        payload = {
            "grant_type": grant_type,
            "code": code
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GET_USER_ACCESS_TOKEN_URL,
                headers=headers,
                data=payload
            )

            print(response.json())

            response_model = UserTokenResponse(**response.json())

            if response_model.code != 0:
                raise LarkBaseHTTPException(
                    code=response_model.code,
                    msg=response_model.message
                )

            return response_model

    async def get_tenant_access_token(self):
        KEY = 'tenant_access_token'

        if self._is_token_still_valid(KEY):
            cached_response_model, _ = self._cache[KEY]
            return cached_response_model

        payload = self._common_auth_payload()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                GET_APP_ACCESS_TOKEN_INTERNAL_URL,
                data=payload
            )

            response_model = TenantTokenResponse(**response.json())

            if response_model.code != 0:
                raise LarkBaseHTTPException(
                    code=response_model.code,
                    msg=response_model.msg
                )
            
            self._cache[KEY] = (
                response_model, int(time.time()) + response_model.expire
            )

            return response_model

    def get_tenant_access_token_sync(self):
        payload = self._common_auth_payload()

        with httpx.Client() as client:
            response = client.post(
                GET_APP_ACCESS_TOKEN_INTERNAL_URL,
                data=payload
            )

            response_model = TenantTokenResponse(**response.json())

            if response_model.code != 0:
                raise LarkBaseHTTPException(
                    code=response_model.code,
                    msg=response_model.msg
                )

            return response_model

    async def get_app_access_token(self) -> AppTokenResponse:
        payload = self._common_auth_payload()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                GET_APP_ACCESS_TOKEN_INTERNAL_URL,
                json=payload
            )

            response_model = AppTokenResponse(**response.json())

            if response_model.code != 0:
                raise LarkBaseHTTPException(
                    response_model.code,
                    response_model.msg
                )

            return response_model

    async def get_user_information(self, user_access_token: str):
        headers = {
            "Authorization": f"Bearer {user_access_token}"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                GET_USER_INFORMATION_URL,
                headers=headers
            )

            response_model = UserInformationResponse(**response.json())

            if response_model.code != 0:
                raise LarkBaseHTTPException(
                    response_model.code,
                    response_model.msg
                )
            
            return response_model
            
    def _is_token_still_valid(self, key: str):
        if key in self._cache:
            _, expires_in = self._cache[key]
            return expires_in > time.time()
        return False

    def _common_auth_payload(self):
        return {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }