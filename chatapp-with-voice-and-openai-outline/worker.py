from openai import OpenAI
import requests

openai_client = OpenAI()



def speech_to_text(audio_binary):

	# Set up Watson Speech-to-Text HTTP Api url
	base_url = 'https://sn-watson-tts.labs.skills.network/text-to-speech/api/v1/voices'
	api_url = base_url+'/speech-to-text/api/v1/recognize'

	# Set up parameters for our HTTP request
	params = {
		'model': 'en-US_Multimedia',
	}

	# Set up the body of our HTTP request
	body = audio_binary

	# Send a HTTP Post request
	response = requests.post(api_url, params=params, data=audio_binary).json()

	# Parse the response to get our transcribed text
	text = 'null'
	while bool(response.get('results')):
		print('speech to text response:', response)
		text = response.get('results').pop().get('alternatives').pop().get('transcript')
		print('recognised text: ', text)
		return text



def text_to_speech(text, voice=""):
    return None


def openai_process_message(user_message):
    return None
