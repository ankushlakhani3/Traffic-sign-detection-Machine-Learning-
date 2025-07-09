from flask import render_template, request

from app import app
from app.model_training import predict_model


@app.route('/')
def home():
    return render_template('index.html', classe_labels=app.config['CLASS_LABELS'])


@app.route('/training_flow')
def training_flow():
    return render_template('training_flow.html')


@app.route('/file_upload')
def file_upload(filename="", filepath=""):
    return render_template("form.html", filename=filename, filepath=filepath)


@app.route('/success', methods=['POST', 'GET'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        file_extension = f.filename.rsplit('.')[-1]
        if file_extension not in app.config['ALLOW_EXTENSION']:
            return file_upload("Sorry the file is not the image.", app.config['404'])
        dest_path = f"static/images/test.{file_extension}"
        save_file_path = app.config['APP_PATH'] / dest_path
        f.save(save_file_path)
        saved_model_path = app.config['APP_PATH'] / 'static/traffic_classifier.h5'
        predicted_label = predict_model(saved_model_path=saved_model_path, image_file_path=save_file_path)
        return file_upload(predicted_label, dest_path)
    return home()
