import httpx
from typing import List, Dict, Optional
from pydantic import BaseModel
from .token_manager import TokenManager
from typing import Literal
from .lark_http_exception import LarkBaseHTTPException

GET_RECORDS_URL = "https://open.larksuite.com/open-apis/bitable/v1/apps/{app_token}/tables/{app_table}/records/{record_id}"
LIST_RECORDS_URL = "https://open.larksuite.com/open-apis/bitable/v1/apps/{app_token}/tables/{app_table}/records"
SEARCH_RECORDS_URL = "https://open.larksuite.com/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/search"
BATCH_UPDATE_URL = "https://open.larksuite.com/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_update"
CREATE_RECORDS_URL = "https://open.larksuite.com/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_create"
DELETE_RECORD_URL = "https://open.larksuite.com/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}"

class SearchResultDataResponse(BaseModel):
    items: list[dict]
    has_more: bool
    total: int


class SearchResultResponse(BaseModel):
    code: int
    msg: str
    data: Optional[SearchResultDataResponse] = None


class ListRecordDataResponse(BaseModel):
    has_more: bool
    page_token: Optional[str] = None
    total: int
    items: List[Dict]


class ListRecordBodyResponse(BaseModel):
    code: int
    msg: str
    data: Optional[ListRecordDataResponse] = None


class BatchUpdateRecordResponse(BaseModel):
    code: int
    msg: str
    data: Optional[Dict] = None


class CreateRecordsDataResponse(BaseModel):
    records: List[dict]


class CreateRecordsBodyResponse(BaseModel):
    code: int
    msg: str
    data: Optional[CreateRecordsDataResponse] = None


class DeletedRecordBodyResponse(BaseModel):
    deleted: bool
    record_id: str


class DeletedRecordResponse(BaseModel):
    code: int
    msg: str
    data: DeletedRecordBodyResponse


class BaseManager:
    def __init__(self, app_token: str, token_manager: TokenManager) -> None:
        self.app_token = app_token
        self._token_manager = token_manager

    async def create_records(
        self,
        table_id: str,
        data: List[dict],
        user_id_type: Literal["union_id", "open_id"] = "union_id"
    ):
        tenant_token = (await self._token_manager.get_tenant_access_token()).tenant_access_token

        formatted_url = CREATE_RECORDS_URL.format(app_token=self.app_token, table_id=table_id)

        async with httpx.AsyncClient() as client:
            payload = {
                "records": data
            }
            
            response = await client.post(
                formatted_url,
                headers={
                    "Authorization": f"Bearer {tenant_token}"
                },
                json=payload,
                params={
                    "user_id_type": user_id_type
                }
            )

            response = CreateRecordsBodyResponse(**response.json())

            if response.code != 0:
                raise LarkBaseHTTPException(response.code, response.msg)

            return response

    async def search_records(
        self,
        table_id: str,
        filter: dict,
        user_id_type: Literal["union_id", "open_id"] = "union_id",
        page_size: int = 100
    ):
        formatted_url = SEARCH_RECORDS_URL.format(
            app_token=self.app_token, table_id=table_id
        )

        tenant_response = await self._token_manager.get_tenant_access_token()

        headers = {"Authorization": f"Bearer {tenant_response.tenant_access_token}"}

        params = {
            "user_id_type": user_id_type,
            "page_size": page_size
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                formatted_url,
                params=params,
                headers=headers,
                json=filter
            )

            return SearchResultResponse(**response.json())

    async def get_records(
        self,
        app_table: str,
        with_shared_url: bool = False,
        page_size: int = 500,
        filter: str = None,
        user_id_type: Literal["open_id", "union_id"] = "union_id",
        all: bool = False,
    ):
        api_token_response = await self._token_manager. \
            get_tenant_access_token()

        headers = {
            "Authorization": f"Bearer {api_token_response.tenant_access_token}"
        }

        formatted_url = LIST_RECORDS_URL.format(
            app_token=self.app_token, app_table=app_table
        )

        params = {
            "page_size": page_size,
            "with_shared_url": with_shared_url,
            "user_id_type": user_id_type,
            "filter": filter
        }

        if all:
            params["has_more"] = True
            params["page_token"] = None
            items = []

            while params["has_more"]:
                print("fetching...")
                print("page_token:", params["page_token"])
                async with httpx.AsyncClient(timeout=None) as client:
                    response = await client.get(
                        formatted_url, params=params, headers=headers
                    )

                    response = response.json()

                    if response["code"] != 0:
                        raise LarkBaseHTTPException(
                            response["code"],
                            response["msg"]
                        )
                    items.extend([*response["data"]["items"]])
                    params["has_more"] = response["data"]["has_more"]
                    params["page_token"] = response["data"]["page_token"] if params["has_more"] else None
            return items
        else:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.get(
                    formatted_url, params=params, headers=headers
                )

                items = response.json()

                return items
    
    async def plain_list_records(
        self,
        app_table: str,
        page_token: str | None = None,
        page_size: int = 500,
        filter: str = None,
        user_id_type: Literal["open_id", "union_id"] = "union_id",
    ):
        tenant_response = await self._token_manager.get_tenant_access_token()

        headers = {
            "Authorization": f"Bearer {tenant_response.tenant_access_token}"
        }

        params = {
            "page_size": page_size,
            "user_id_type": user_id_type,
            "filter": filter
        }
        
        formatted_url = LIST_RECORDS_URL.format(
            app_token=self.app_token,
            app_table=app_table
        )

        if page_token:
            params["page_token"] = page_token

        async with httpx.AsyncClient(timeout=None) as http:
            response = await http.get(
                formatted_url,
                headers=headers,
                params=params
            )

            return ListRecordBodyResponse(**response.json())

    def plain_list_records_sync(
        self,
        app_table: str,
        page_token: str | None = None,
        page_size: int = 500,
        filter: str = None,
        user_id_type: Literal["open_id", "union_id"] = "union_id",
    ):
        tenant_response = self._token_manager.get_tenant_access_token_sync()

        headers = {
            "Authorization": f"Bearer {tenant_response.tenant_access_token}"
        }

        params = {
            "page_size": page_size,
            "user_id_type": user_id_type,
            "filter": filter
        }
        
        formatted_url = LIST_RECORDS_URL.format(
            app_token=self.app_token,
            app_table=app_table
        )

        if page_token:
            params["page_token"] = page_token

        with httpx.Client(timeout=None) as http:
            response = http.get(
                formatted_url,
                headers=headers,
                params=params
            )

            return ListRecordBodyResponse(**response.json())

    async def update_records(
        self,
        table_id: str,
        records: dict
    ):
        formatted_url = BATCH_UPDATE_URL.format(
            app_token=self.app_token,
            table_id=table_id
        )

        tenant_response = await self._token_manager.get_tenant_access_token()

        headers = {
            "Authorization": f"Bearer {tenant_response.tenant_access_token}"
        }

        async with httpx.AsyncClient() as http:
            response = await http.post(
                formatted_url,
                headers=headers,
                json=records
            )

            data = BatchUpdateRecordResponse(**response.json())

            if data.code != 0:
                raise LarkBaseHTTPException(data.code, data.msg)
            
            return data

    
    async def delete_record(
        self, 
        table_id: str, 
        record_id: str
    ):
        formatted_url = DELETE_RECORD_URL.format(
            app_token=self.app_token,
            table_id=table_id,
            record_id=record_id
        )

        tenant_token = (await self._token_manager.get_tenant_access_token()).tenant_access_token

        headers = {
            "Authorization": f"Bearer {tenant_token}"
        }

        async with httpx.AsyncClient() as client:
            response = await client.put(
                formatted_url, 
                headers=headers
            )

            data = DeletedRecordResponse(**response.json())

            if data.code != 0:
                raise LarkBaseHTTPException(
                    code=data.code,
                    msg=data.msg
                ) 
            
            return data