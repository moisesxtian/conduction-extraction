import os
import httpx

async def upload_image(
    image: bytes, 
    access_token: str, 
    image_type="message"
):
    """
    Uploads an image to the specified API and returns the image_key.

    Args:
        image_path (str | bytes): The file path or binary of the image to be uploaded.
        access_token (str): The tenant access token for authorization.
        image_type (str, optional): The type of image. Defaults to "message".
                                     Other possible value: "avatar".

    Returns:
        str: The image_key if the upload is successful.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image size exceeds 10MB.
        Exception: If the API response indicates a failure.
    """
    # Constants
    MAX_SIZE_MB = 10
    URL = "https://open.larksuite.com/open-apis/image/v4/put/"

    # Check if file exists
    # if not os.path.isfile(image):
    #     raise FileNotFoundError(f"The file '{image_path}' does not exist.")

    # # Check file size
    # file_size = os.path.getsize(image_path)
    # if file_size > MAX_SIZE_MB * 1024 * 1024:
    #     raise ValueError(f"Image size exceeds {MAX_SIZE_MB}MB limit.")

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    # Prepare multipart/form-data
    async with httpx.AsyncClient() as client:
        # with open(image_path, 'rb') as image_file:
        files = {
            "image": image,
        }
        
        data = {
            "image_type": image_type
        }

        try:
            response = await client.post(URL, headers=headers, files=files, data=data)
        except httpx.RequestError as e:
            raise Exception(f"HTTP request failed: {e}")

    # Check HTTP response status
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    # Parse JSON response
    try:
        response_json = response.json()
    except ValueError:
        raise Exception("Failed to parse JSON response.")

    # Check API response code
    if response_json.get("code") != 0:
        msg = response_json.get("msg", "Unknown error")
        raise Exception(f"API error {response_json.get('code')}: {msg}")

    # Extract image_key
    image_key = response_json.get("data", {}).get("image_key")
    if not image_key:
        raise Exception("image_key not found in the response.")

    return image_key