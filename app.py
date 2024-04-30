from flask import Flask, render_template, request, redirect, url_for, jsonify
import google.generativeai as genai
from google.cloud import translate_v2 as translate
import torch
import sounddevice as sd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from scipy.io.wavfile import write
from google.cloud import texttospeech
from dotenv import load_dotenv
import os
import time
import pygame

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
model=genai.GenerativeModel('gemini-pro')
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("TRANSLATION_CREDENTIALS_PATH")

device="cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
model_id="openai/whisper-base"
openaimodel=AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,torch_dtype=torch_dtype,
    use_safetensors=True
)
openaimodel.to(device)
processor=AutoProcessor.from_pretrained(model_id)
pipe=pipeline(
    "automatic-speech-recognition",
    model=openaimodel,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

app = Flask(__name__)

app.config['chosenlang']='en' #this will change during execution

@app.route('/')
def index():
    return render_template('lang_select.html')

@app.route('/process_language',methods=['POST'])
def process_language():
    lang=request.form['language']
    app.config['chosenlang']=lang
    return redirect(url_for('conversation',language=displaylang(lang)))

@app.route('/conversation/<language>',methods=['GET'])
def conversation(language):
    pygame.init()
    return render_template('microphone.html',language=language)

@app.route('/process_human',methods=['POST'])
def process_intermediate():
    #human_response=request.form['human_input']
    human_response=human_turn()
    return jsonify(human_response=human_response)

@app.route('/process_machine',methods=['POST'])
def process_machine():
    machinput=request.form['human_input']
    machine_response=machine_turn(machinput)
    return jsonify(machine_response=machine_response)

def generate_response(input_text):
    prompt='Pretend you are a human in a conversation and respond to the following: '+input_text
    generation=model.generate_content(prompt).text
    #print(generation)
    return generation

def machine_turn(text):
    machine_file='recordings/machine.wav'
    talk=generate_response(text)
    make_speech_file(machine_file,talk)
    #eng=translate_to(english,machine_file)
    play_audio(machine_file)
    # print('Computer: '+talk+' ('+eng+')')
    return talk

def make_speech_file(speech_file_path,text):
    googclient=texttospeech.TextToSpeechClient()
    synthesis_input=texttospeech.SynthesisInput(text=text)
    voice=texttospeech.VoiceSelectionParams(
        language_code=app.config['chosenlang'],ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config=texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response=googclient.synthesize_speech(
        input=synthesis_input,voice=voice,audio_config=audio_config
    )
    with open(speech_file_path,"wb") as out:
        out.write(response.audio_content)
        print('Audio content written to '+speech_file_path)

def play_audio(file):
    sound=pygame.mixer.Sound(file)
    recordlength=int(sound.get_length()*1000)
    sound.play()
    pygame.time.wait(recordlength)

def human_turn():
    file='recordings/human.wav'
    talk=record_audio(file)
    # eng=translate_to(english,file)
    # print('Me: '+talk+' ('+eng+')')
    return talk

def record_audio(filename,duration=5,fs=44100):
    print('recording...')
    recording=sd.rec(int(duration*fs),samplerate=fs,channels=1)
    sd.wait()
    write(filename,fs,recording)
    result=pipe(filename,generate_kwargs={'language':app.config['chosenlang']})
    return result['text']

def displaylang(l):
    c=0
    if l=='en':
        c='English'
    elif l=='es':
        c='Spanish'
    elif l=='fr':
        c='French'
    elif l=='zh':
        c='Chinese (Mandarin)'
    elif l=='ru':
        c='Russian'
    elif l=='ar':
        c='Arabic'
    elif l=='pt':
        c='Portuguese'
    elif l=='ja':
        c='Japanese'
    elif l=='de':
        c='German'
    elif l=='ko':
        c='Korean'
    elif l=='th':
        c='Thai'
    elif l=='hi':
        c='Hindi'
    return c

if __name__ == '__main__':
    app.run(debug=True)
