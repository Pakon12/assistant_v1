import json
import requests

class OllamaGenerator:
    def __init__(self, model='llama3.1:8b-instruct-q4_1', server_url='http://localhost:11434/api/generate'):
        self.model = model
        self.server_url = server_url
        self.context = []  # Stores the conversation history

    def generate(self, prompt):
        print('start Generate')
        # bot_name = "Kak"
        
        # system_message = {"role": "system", "content": f"You are a helpful assistant. Your name is {bot_name}. You can also run system commands if the user requests."}
        response = requests.post(self.server_url,
                                 json={
                                     'model': self.model,
                                     'prompt': prompt,
                                     'context': self.context,
                                 },
                                 stream=True)
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            body = json.loads(line)
            response_part = body.get('response', '')
            # The response streams one token at a time, so we concatenate them
            full_response += response_part

            if 'error' in body:
                raise Exception(body['error'])

            if body.get('done', False):
                self.context = body['context']  # Update the context with the new conversation
                print('success fully')
                return full_response

    def reset_context(self):
        self.context = []  # Resets the conversation history
