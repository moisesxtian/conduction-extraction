import json
import httpx
from pydantic import BaseModel
from dtos.typings import PlateGCNotification, ClientRecord



async def send_notification(data: PlateGCNotification):
    # url = "https://open.larksuite.com/open-apis/bot/v2/hook/0ff11ca5-2201-429f-adb5-d084f55875cf" # test-notif
    url = "https://open.larksuite.com/open-apis/bot/v2/hook/0a318b33-68ae-4a7a-b7bb-22a3bfc905bc" # official

    image_key = data.image_key

    header_content = f"{data.status} PLATE | STICKER DETECTED!"
    content = f"Detected: **{data.plate}**\n"
    client_info = ""
    template_color = "red" if data.status == "POSITIVE" else "yellow"

    if isinstance(data.clients, ClientRecord):
        content += f"Client: **{data.clients.plate_number}** [**{data.clients.car_model}** - **{data.clients.client}**]\n"

        client_info += f"\nClient Information:\n- Plate | Sticker: **{data.clients.plate_number}**\n" \
                    f"- Vehicle: {data.clients.car_model}\n" \
                    f"- Client Name: **{data.clients.client}**\n" \
                    f"- Endorsement Date: {data.clients.endo_date}\n" \
                    f"- CH Code: {data.clients.ch_code}\n"
    elif isinstance(data.clients, list):
        for client in data.clients:
            content += f"Client: **{client.plate_number}** [**{client.car_model}** - **{client.client}**]\n"

            client_info += f"\nClient Information:\n- Plate | Sticker: **{client.plate_number}**\n" \
                        f"- Vehicle: {client.car_model}\n" \
                        f"- Client Name: **{client.client}**\n" \
                        f"- Endorsement Date: {client.endo_date}\n" \
                        f"- CH Code: {client.ch_code}\n"

    content += client_info


    card_data = {
        "config": {
            "wide_screen_mode": True
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "content": content,
                    "tag": "lark_md"
                }
            },
            {
                "tag": "img",
                "img_key": image_key,
                "alt": {
                    "tag": "plain_text",
                    "content": ""
                },
                "mode": "fit_horizontal",
                "preview": True
            }
        ],
        "header": {
            "template": template_color,
            "title": {
                "content": header_content,
                "tag": "plain_text"
            }
        }
    }

    body = {"msg_type": "interactive", "card": card_data}
    headers = {"Content-Type": "application/json"}

    print('notif_body', body)
    

    async with httpx.AsyncClient() as client:
        response = await client.post(url=url, json=body, headers=headers)

        if response.status_code == 200:
            print("Message sent successfully!")
        else:
            print(f"Failed to send message. Status code: {response.status_code}, Response: {response.text}")