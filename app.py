from flask import Flask, render_template, request, redirect, url_for

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

if __name__ == '__main__':
    app.run(debug=True)
