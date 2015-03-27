import os, sys

sys.path.insert(1, os.path.join(os.path.abspath('.'), 'venv/Lib/site-packages'))

from flask import Flask, request, redirect, url_for, jsonify, render_template, send_from_directory, abort
from werkzeug.utils import secure_filename
from flask.ext.cors import CORS, cross_origin

if not os.path.exists("uploads"):
    os.makedirs("uploads")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# cors = CORS(app, resources={r'/api/*': {"origins": "*"}}, allow_headers='Content-Type')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
def hello():
    return "Hello world!"

@cross_origin()
@app.route("/api/upload", methods=['POST'])
def upload_image():
	file = request.files['file']
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		return jsonify(filename=filename)
	else:
		abort(400)
            #return redirect(url_for('uploaded_file', filename=filename))

@cross_origin()
@app.route("/api/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
	
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


if __name__ == "__main__":
    app.run(debug=True)
