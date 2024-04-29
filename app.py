from flask import Flask, render_template, request, redirect, url_for, jsonify
import google.generativeai as genai
from google.cloud import translate_v2 as translate
from dotenv import load_dotenv
import os
import time

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
model=genai.GenerativeModel('gemini-pro')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('lang_select.html')

@app.route('/process_language',methods=['POST'])
def process_language():
    lang=request.form['language']
    return redirect(url_for('conversation',language=lang))

@app.route('/conversation/<language>',methods=['GET'])
def conversation(language):
    return render_template('microphone.html',language=language)

@app.route('/process_human',methods=['POST'])
def process_intermediate():
    human_response=request.form['human_input']
    return jsonify(human_response=human_response)

@app.route('/process_machine',methods=['POST'])
def process_machine():
    machinput=request.form['human_input']
    machine_response=machine_turn(machinput)
    return jsonify(machine_response=machine_response)

def generate_response(input_text):
    prompt='Please respond logically to the following sentence in a conversation: '+input_text
    generation=model.generate_content(prompt).text
    #print(generation)
    return generation

def machine_turn(text):
    machine_file='recordings/machine.wav'
    talk=generate_response(text)
    # make_speech_file(machine_file,talk)
    # eng=translate_to(english,machine_file)
    # play_audio(machine_file)
    # print('Computer: '+talk+' ('+eng+')')
    return talk

if __name__ == '__main__':
    app.run(debug=True)
