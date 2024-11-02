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
import time
from openai import OpenAI
import speech_recognition as sr
import scipy.io.wavfile as wav
import whisper
from pathlib import Path

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
client=OpenAI()

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
    if os.path.exists('recordings/human.wav'):
        os.remove('recordings/human.wav')
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

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error':'no audio file'}),400
    try:
        audiofile=request.files['audio']
        filename='recordings/human.wav'
        os.makedirs('recordings',exist_ok=True)
        if os.path.exists(filename):
            os.remove(filename)
        audiofile.save(filename)
        if not os.path.exists(filename):
            return jsonify({'error':'file not saved properly'}),500
        try:
            from pydub import AudioSegment
            audio=AudioSegment.from_wav(filename)
            audio.export(filename, format='wav', parameters=['-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1']) #line copied in from Claude
        except Exception as e:
            print(f'Error converting audio: {str(e)}')
        return jsonify({'success':True})
    except Exception as e:
        print(f'Error processing audio: {str(e)}')
        return jsonify({'error':False}),500
    # if 'audio' not in request.files:
    #     return jsonify({'error': 'No audio file'}), 400
    #
    # try:
    #     audiofile = request.files['audio']
    #     filename = 'recordings/human.wav'
    #
    #     # Save the file
    #     audiofile.save(filename)
    #
    #     # Verify the file was saved and is readable
    #     if not os.path.exists(filename):
    #         return jsonify({'error': 'File not saved properly'}), 500
    #
    #     # Convert to correct format if needed
    #     try:
    #         from pydub import AudioSegment
    #         audio = AudioSegment.from_wav(filename)
    #         audio.export(filename, format='wav', parameters=['-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1'])
    #     except Exception as e:
    #         print(f"Error converting audio: {str(e)}")
    #
    #     transcription = speechtotext(filename)
    #     return jsonify({'transcription': transcription})
    # except Exception as e:
    #     print(f"Error processing audio: {str(e)}")
    #     return jsonify({'error': str(e)}), 500

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
    while not os.path.exists(file):
        time.sleep(0.1)
    talk=record_audio(file)
    updatetrans(talk,True)
    print('human response: ',talk)
    # eng=translate_to(english,file)
    # print('Me: '+talk+' ('+eng+')')
    return talk

def speechtotext(filename):
    # recognizer=sr.Recognizer()
    # with sr.AudioFile(filename) as source:
    #     audio=recognizer.record(source)
    # try:
    #     text=recognizer.recognize_google(audio)
    #     return text
    # except sr.UnknownValueError:
    #     return 'Could not understand audio'
    # except sr.RequestError as e:
    #     return 'Could not request results from Google Speech Recognition service'
    """
    Convert speech to text using OpenAI's Whisper model.

    Args:
        filename (str): Path to the audio file

    Returns:
        str: Transcribed text or error message
    """
    # try:
    #     # Check if file exists
    #     if not Path(filename).is_file():
    #         return "Error: Audio file not found"
    #
    #     # Load the model (first run will download the model)
    #     model = whisper.load_model("base")  # Other options: "tiny", "small", "medium", "large"
    #
    #     # Transcribe audio
    #     result = model.transcribe(filename)
    #
    #     return result["text"].strip()
    #
    # except Exception as e:
    #     return f"Error during transcription: {str(e)}"
    audio_file=open(filename, "rb")
    transcription = client.audio.transcriptions.create(
      model="whisper-1",
      file=audio_file
    )
    print(transcription.text)
    return transcription.text

def convertaudioforwhisper(input_file, output_file=None):
    """
    Convert audio file to format compatible with Whisper API.
    Args:
        input_file: Path to input audio file
        output_file: Path to save converted file (optional)
    Returns:
        Path to converted audio file
    """
    from pydub import AudioSegment

    if output_file is None:
        output_file = input_file

    try:
        # Load audio file
        audio = AudioSegment.from_wav(input_file)

        # Convert to proper format
        # - Sample width of 2 bytes (16 bit)
        # - Sample rate of 16kHz
        # - Single channel (mono)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

        # Export with specific parameters
        audio.export(
            output_file,
            format="wav",
            parameters=[
                "-ac", "1",  # mono
                "-ar", "16000",  # 16kHz
                "-sample_fmt", "s16",  # 16-bit
                "-acodec", "pcm_s16le"  # PCM signed 16-bit little-endian
            ]
        )
        return output_file
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        return None

def record_audio(filename,duration=5,fs=44100):
    # print('recording...')
    # recording=sd.rec(int(duration*fs),samplerate=fs,channels=1)
    # sd.wait()
    # normalized=np.int16(recording*32767)
    # wav.write(filename,fs,normalized)
    converted=convertaudioforwhisper(filename)
    return speechtotext(converted)

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