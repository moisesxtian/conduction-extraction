import os
import requests
import json

def upload_image(image_path, access_token, image_type="message"):
    """
    Uploads an image to the specified API and returns the image_key.

    Args:
        image_path (str): The file path to the image to be uploaded.
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
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The file '{image_path}' does not exist.")

    # Check file size
    file_size = os.path.getsize(image_path)
    if file_size > MAX_SIZE_MB * 1024 * 1024:
        raise ValueError(f"Image size exceeds {MAX_SIZE_MB}MB limit.")

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    # Prepare multipart/form-data
    with open(image_path, 'rb') as image_file:
        files = {
            "image": (os.path.basename(image_path), image_file, "application/octet-stream"),
        }
        data = {
            "image_type": image_type
        }

        try:
            response = requests.post(URL, headers=headers, files=files, data=data)
        except requests.exceptions.RequestException as e:
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

def send_image_to_group_chat(access_token, chat_id, image_key):
    """
    Sends an image to a specified group chat.

    Args:
        access_token (str): The tenant access token for authorization.
        chat_id (str): The ID of the chat where the image will be sent.
        image_key (str): The image key of the uploaded image.

    Returns:
        dict: The response from the API if the message was successfully sent.

    Raises:
        Exception: If there is any issue with the API request.
    """
    # API endpoint
    url = "https://open.larksuite.com/open-apis/im/v1/messages"

    # Request headers
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json; charset=utf-8'
    }

    # Request body
    data = {
        "receive_id": chat_id,
        "msg_type": "image",
        "content": json.dumps({
            "image_key": image_key
        })
    }

    # Query parameter for receive_id_type
    params = {
        "receive_id_type": "chat_id"
    }

    # Send the POST request
    try:
        response = requests.post(url, headers=headers, params=params, data=json.dumps(data))
        response.raise_for_status()  # Check for HTTP errors
        return response.json()  # Return the API response as a JSON dictionary
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to send image: {e}")


# if __name__ == "__main__":
#     # Replace with your actual access token and image path
#     ACCESS_TOKEN = "t-g2059j8g7K7O5ARFOEDRTQIWDNYNJHUPJKRLG5JL"
#     IMAGE_PATH = "/Users/julius/Downloads/alpr-dev 2/images/AHA8200.jpg"
#     CHAT_ID = "oc_3f40eb6f785a0cd7a22c1f879e9e8f2c"

#     try:
#         key = upload_image(IMAGE_PATH, ACCESS_TOKEN, image_type="message")
#         print(f"Image uploaded successfully. Image Key: {key}")

#         try:
#             response = send_image_to_group_chat(ACCESS_TOKEN, CHAT_ID, key)
#             print(f"Image sent successfully: {response}")
#         except Exception as e:
#             print(f"Error: {e}")
#     except Exception as e:
#         print(f"Failed to upload image: {e}")



