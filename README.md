# Google Hackathon - Foreign Language Practice Assistant

Language learning apps like Duolingo are great for early stage learning, but can fall short when someone wants to reach fluency. This leaves a gray area individuals can end up in where they have outgrown the language learning apps, but are still not ready yet to effectively converse with real native speakers in a non-practice setting. 

This project is meant to address the issue by letting users have a verbal conversation with an AI bot in their chosen language. While there are obvious downsides to this, using an AI language model gives the user a playground so to speak where they can practice speech with a dummy conversation. 

How to use:
1. Select your language (be aware that the less widely spoken languages will not have as well-trained models, and may give weird output at times).
2. Click "Record" and you will have 5 seconds to say what you want to the machine. After this the machine will resond, then you click record again and so on.
3. After the conversation ends the user can download a transcript of the conversation text to their device, in case there are any new words/phrases they would like to remember.

How it works:
- Google Gemini API is used to craft responses to the user input text. Google Vertex AI is used specifically so I could give a system instruction and so that chat history is remembered througout the conversation
- Google Translate API is used to translate into and from English
- OpenAI Whisper is used to convert the user's speech to text
- Audio is played with Pygame

Features to add in the future:
- Alerts for grammatical mistakes
- Accent feedback and critiques
- Click text to show translation in English
- More upscaled UI

# Steps to run:
1. Clone the repository
2. Create a .env file with the environment variables from the .env.template file (this will require a Google Cloud account)
3. Install requirements by running `pip install -r requirements.txt` in terminal
4. Run `python app.py` to start the server
5. Visit the localhost link shown in terminal
