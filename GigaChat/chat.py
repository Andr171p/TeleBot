import requests
import base64
import uuid
import json


class GigaChat:
    def __init__(self, client_id, client_secret, prompt_file_path=None):
        self.credentials = f"{client_id}:{client_secret}"
        self.auth_token = None
        self.prompt_file_path = prompt_file_path

    def get_auth_token(self):
        encoded_credentials = base64.b64encode(self.credentials.encode("utf-8")).decode("utf-8")
        self.auth_token = encoded_credentials

    def get_giga_api(self, scope="GIGACHAT_API_PERS"):
        self.get_auth_token()
        rqUID = str(uuid.uuid4())
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": rqUID,
            "Authorization": f"Basic {self.auth_token}"
        }

        payload = {
            "scope": scope
        }
        try:
            response = requests.post(url, headers=headers, data=payload, verify=False)
            return response.json()["access_token"]
        except requests.RequestException as _ex:
            print(f"[ERROR] : {_ex}")
            return -1

    def get_models(self):
        giga_api = self.get_giga_api()
        url = "https://gigachat.devices.sberbank.ru/api/v1/models"
        payload = {}
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {giga_api}"
        }

        response = requests.request("GET", url, headers=headers, data=payload, verify=False)

        return response.text

    def get_chat_answer(self, user_message):
        giga_api = self.get_giga_api()
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        payload = json.dumps({
            "model": "GigaChat",
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "temperature": 0.5,
            "top_p": 0.1,
            "n": 1,
            "stream": False,
            "max_tokens": 512,
            "repetition_penalty": 1,
            "update_interval": 0
        })
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {giga_api}"
        }

        try:
            response = requests.request("POST", url, headers=headers,
                                        data=payload, verify=False)
            # return display(Markdown(response.json()["choices"][0]["message"]["content"]))
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as _ex:
            print(f"[ERROR] : {_ex}")
            return -1

    def add_prompt(self):
        with open(self.prompt_file_path, "r",
                  encoding="utf-8") as file:
            prompt = file.readline()

        conversation_history = [{
            "role": "system",
            "content": prompt
        }]

        return conversation_history

    def get_chat_dialog(self, user_message, conversation_history=None):
        giga_api = self.get_giga_api()
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        if conversation_history is None:
            conversation_history = []

        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        payload = json.dumps({
            "model": "GigaChat:latest",
            "messages": conversation_history,
            "temperature": 0.5,
            "top_p": 0.1,
            "n": 1,
            "stream": False,
            "max_tokens": 512,
            "repetition_penalty": 1,
            "update_interval": 0
        })
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {giga_api}"
        }

        try:
            response = requests.post(url, headers=headers, data=payload, verify=False)
            response_data = response.json()

            print(response_data)

            conversation_history.append({
                "role": "assistant",
                "content": response_data["choices"][0]["message"]["content"]
            })
            return response, conversation_history
        except requests.RequestException as _ex:
            print(f"[ERROR] : {_ex}")
            return None, conversation_history



