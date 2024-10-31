from flask import Flask, render_template, request
from models import ProtoNet
import os
import io

import base64

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
#app.config['DOWNLOAD_FOLDER'] = 'downloads'

model = ProtoNet()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['GET', 'POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        saveLocation = f.filename
        f.save(saveLocation)
        data, inference, confidence, exec_time = model.infer(saveLocation)

        buf = io.BytesIO()
        data.save(buf, format='JPEG')
        jpg_as_text = base64.b64encode(buf.getvalue()).decode('utf-8')

        # delete file after making an inference
        os.remove(saveLocation)
        # respond with the inference
        return render_template('inference.html', name=inference, confidence=confidence, exec_time=exec_time, photo=jpg_as_text)


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)