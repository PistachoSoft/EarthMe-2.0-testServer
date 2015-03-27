import os
#import sys

#sys.path.insert(1, os.path.join(os.path.abspath('.'), 'venv/Lib/site-packages'))

import time
from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from werkzeug.utils import secure_filename
from flask.ext.cors import CORS
import EarthMe2

if not os.path.exists("uploads"):
    os.makedirs("uploads")

if not os.path.exists("processed"):
    os.makedirs("processed")

UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
cors = CORS(app, resources=r'/api/*', allow_headers='Content-Type')
>>>>>>> 67937bbe672c7fa28335b6264e703d28744bfb20

# system's time, uuid attempt
current_milli_time = lambda: int(round(time.time() * 1000))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

	
@app.route("/api/upload", methods=['POST'])
def upload_image():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = str(current_milli_time()) + "_" + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        if EarthMe2.process_img(filename,
                                img_dir=app.config['UPLOAD_FOLDER'],
                                output_dir=app.config['PROCESSED_FOLDER']):

            return jsonify(filename=filename)
        else:
            abort(500)
    else:
        abort(400)
        # return redirect(url_for('uploaded_file', filename=filename))


@cross_origin()
@app.route("/api/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'],
                               filename)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


if __name__ == "__main__":
    EarthMe2.setup(gen_palette=True)
    app.run()
