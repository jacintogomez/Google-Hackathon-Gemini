import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import sounddevice as sd
from scipy.io.wavfile import write
from google.cloud import texttospeech
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
from dotenv import load_dotenv
import os
import pygame
import base64
import json
import speech_recognition as sr
import scipy.io.wavfile as wav

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
# model=genai.GenerativeModel('gemini-pro')
# genai.configure(api_key=GOOGLE_API_KEY)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("TRANSLATION_CREDENTIALS_PATH")

credpath=os.getenv("TRANSLATION_CREDENTIALS_PATH")
decode=base64.b64decode(os.getenv("TRANSLATION_CREDENTIALS_ENCODED"))
jsondecode=json.loads(decode)
with open(credpath,'wb') as file:
    file.write(decode)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=credpath

project_id=os.getenv("CLOUD_PROJECT_ID")
location=os.getenv("CLOUD_PROJECT_LOCATION")
vertexai.init(project=project_id, location=location)
model=GenerativeModel(
    model_name="gemini-1.0-pro-002",
    system_instruction=[
        "You are a human in a voice conversation with someone, keep the conversation going. Give a brief response of about 1-2 sentences in the same language that the input is given in. No emojis since this is verbal",
    ],
)
chat=model.start_chat(response_validation=False)

# device="cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
# model_id="openai/whisper-base"
# openaimodel=AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id,torch_dtype=torch_dtype,
#     use_safetensors=True
# )
# openaimodel.to(device)
# processor=AutoProcessor.from_pretrained(model_id)
# pipe=pipeline(
#     "automatic-speech-recognition",
#     model=openaimodel,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,
#     return_timestamps=True,
#     torch_dtype=torch_dtype,
#     device=device,
# )

app = Flask(__name__)

app.config['chosenlang']='en' #this will change during execution
app.config['temp_msg']='Hello, how are you today?' #this too
app.config['transpath']='session/trans.txt'

@app.route('/')
def index():
    return render_template('lang_select.html')

@app.route('/process_language',methods=['POST'])
def process_language():
    lang=request.form['language']
    app.config['chosenlang']=lang
    return redirect(url_for('conversation',language=lang))

@app.route('/conversation/<language>',methods=['GET'])
def conversation(language):
    cleartranscript()
    pygame.init()
    return render_template('microphone.html',language=displaylang(language))

@app.route('/process_human',methods=['POST'])
def process_intermediate():
    #human_response=request.form['human_input']
    human_response=human_turn()
    app.config['temp_msg']=human_response
    return jsonify(human_response=human_response)

@app.route('/process_machine',methods=['POST'])
def process_machine():
    machinput=app.config['temp_msg']
    machine_response=machine_turn(machinput)
    print('machine response: ',machine_response)
    return jsonify(machine_response=machine_response)

@app.route('/machine_speak')
def machine_speak():
    mf='recordings/machine.wav'
    play_audio(mf)
    return 'Audio playing'

@app.route('/session/trans.txt')
def download_transcript():
    tpath=app.config['transpath']
    if os.path.exists(tpath):
        return send_file(tpath,as_attachment=True)

@app.route('/stop_session')
def stop_session():
    pygame.quit()
    return 'Pygame session stopped'

def get_chat_response(chat: ChatSession,prompt: str) -> str:
    text_response=[]
    print('prompt: '+prompt)
    responses=chat.send_message(prompt,stream=True)
    print('responses: ',responses,type(responses))
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

def generate_response(input_text):
    print('chat: ',chat,type(chat))
    generation=get_chat_response(chat,input_text)
    #print(generation)
    return generation

def machine_turn(text):
    machine_file='recordings/machine.wav'
    talk=generate_response(text)
    updatetrans(talk,False)
    make_speech_file(machine_file,talk)
    #eng=translate_to(english,machine_file)
    #play_audio(machine_file)
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
    updatetrans(talk,True)
    print('human response: ',talk)
    # eng=translate_to(english,file)
    # print('Me: '+talk+' ('+eng+')')
    return talk

def speechtotext(filename):
    recognizer=sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio=recognizer.record(source)
    try:
        text=recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return 'Could not understand audio'
    except sr.RequestError as e:
        return 'Could not request results from Google Speech Recognition service'

def record_audio(filename,duration=5,fs=44100):
    print('recording...')
    recording=sd.rec(int(duration*fs),samplerate=fs,channels=1)
    sd.wait()
    normalized=np.int16(recording*32767)
    wav.write(filename,fs,normalized)
    return speechtotext(filename)

# def record_audio(filename):
#     duration=5
#     sample_rate=44100
#     channels=1
#     print('Recording...')
#     audio_data=sd.rec(int(duration*sample_rate),samplerate=sample_rate,channels=channels)
#     sd.wait()
#     print('Recording finished')
#     with wave.open(filename,'wb') as wf:
#         wf.setnchannels(channels)
#         wf.setsampwidth(2)
#         wf.setframerate(sample_rate)
#         wf.writeframes(audio_data.tobytes())
#     recognizer=sr.Recognizer()
#     try:
#         with sr.AudioFile(filename) as source:
#             audio=recognizer.record(source)
#         text=recognizer.recognize_google(audio)
#         return text
#     except sr.UnknownValueError:
#         return 'Could not understand audio'
#     except sr.RequestError as e:
#         return 'Could not request results from Google Speech Recognition service'

def updatetrans(line,isme):
    pre=''
    if isme:
        pre='Me: '
    else:
        pre='Machine: '
    with open(app.config['transpath'],'a') as file:
        file.write(pre+line+'\n'+'\n')
    print('Updated transcript file')

def cleartranscript():
    open(app.config['transpath'],'w').close()

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