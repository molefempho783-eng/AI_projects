# worker.py - corrected version
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
import requests

# Watsonx configuration
PROJECT_ID = "skills-network"
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

model_id = "mistralai/mistral-medium-2505"  # verify this model ID is correct

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)


def watsonx_process_message(user_message):
    # Proper indentation: 4 spaces inside function
    prompt = f"""
    Translate the following English sentence into Spanish. 
    Reply ONLY with the translation, no explanations, no formatting, no extra text.

    English: {user_message}
    Spanish:
    """
    response_text = model.generate_text(prompt=prompt)
    print("watsonx response:", response_text)
    return response_text.strip()


def text_to_speech(text, voice=""):
    # Watson Text-to-Speech HTTP API URL
    base_url = 'https://sn-watson-tts.labs.skills.network'
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'

    if voice != "" and voice != "default":
        api_url += "&voice=" + voice

    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }

    json_data = {
        'text': text,
    }

    response = requests.post(api_url, headers=headers, json=json_data)
    print('Text-to-Speech response:', response)
    return response.content


def speech_to_text(audio_binary):
    # REPLACE the placeholder URL with the actual Speech-to-Text endpoint
    base_url = 'https://sn-watson-stt.labs.skills.network'   # corrected example URL
    api_url = base_url + '/speech-to-text/api/v1/recognize'

    params = {
        'model': 'en-US_Multimedia',
    }

    response = requests.post(api_url, params=params, data=audio_binary).json()

    # Safely extract transcribed text
    text = 'null'
    if response.get('results'):
        text = response['results'][0]['alternatives'][0]['transcript']
        print('recognised text:', text)
    else:
        print('No speech recognized')
    return text