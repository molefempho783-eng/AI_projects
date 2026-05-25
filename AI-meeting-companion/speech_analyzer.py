import torch
import gradio as gr
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# ---------- Watsonx LLM setup ----------
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

params = {
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.1,
}

LLAMA_model = Model(
    model_id='meta-llama/llama-3-2-11b-vision-instruct',
    credentials=my_credentials,
    params=params,
    project_id="skills-network",
)

llm = WatsonxLLM(LLAMA_model)

# ---------- Prompt template (Llama 3 chat format) ----------
temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""
pt = PromptTemplate(input_variables=["context"], template=temp)
prompt_chain = LLMChain(llm=llm, prompt=pt)

# ---------- Speech-to-text + summarization ----------
def transcript_and_summarize(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    transcript = pipe(audio_file, batch_size=8)["text"]
    summary = prompt_chain.run(transcript)
    return summary

# ---------- Gradio interface ----------
audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(
    fn=transcript_and_summarize,
    inputs=audio_input,
    outputs=output_text,
    title="AI Meeting Companion",
    description="Upload a meeting audio file. The app will transcribe it and extract key points & details."
)

iface.launch(server_name="0.0.0.0", server_port=7860)