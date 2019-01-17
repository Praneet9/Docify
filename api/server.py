from flask import Flask, request, jsonify
import os
from processing import recognise_text, seven_segment, _init_model, get_labels_from_aadhar, get_labels_from_licence
from cheque_details_extraction import get_micrcode, ensemble_acc_output, ensemble_ifsc_output
import datetime
import db
from face_matching import match_faces

app = Flask(__name__)
UPLOAD_FOLDER = './UPLOAD_FOLDER/'

_init_model()

@app.route('/image/upload', methods=['POST'])
def index():
    
    if request.method == 'POST':

        current_time = str(datetime.datetime.now())

        image_type = request.form['type']
        
        filename = UPLOAD_FOLDER + image_type + '/' + current_time + '.png'

        if not os.path.exists(UPLOAD_FOLDER + image_type):
            os.mkdir(UPLOAD_FOLDER + image_type)
            os.mkdir(UPLOAD_FOLDER + image_type + '/' + 'faces')
        
        if image_type == 'Bank Cheque':
            details = {}
            photo = request.files['photo']
            photo.save(filename)
            details['MICR'] = get_micrcode(filename)
            details['ACC.No'] = ensemble_acc_output(filename)
            details['IFSC'] = ensemble_ifsc_output(filename)

            return jsonify({'status':True, 'fields': details, 'image_path': filename, 'photo_path': 'none' })

        elif image_type == 'Seven Segment':
            details = {}
            photo = request.files['photo']
            photo.save(filename)

            text = seven_segment(filename)

            details[0] = text

            return jsonify({'status':True, 'fields': details, 'image_path': filename, 'photo_path': 'none' })

        else:
            photo_path = UPLOAD_FOLDER + image_type + '/' + 'faces' + '/' + current_time + '.png'
            
            photo = request.files['photo']
            photo.save(filename)

            data, photo_path = recognise_text(filename, photo_path)
            #details = { idx : text for idx, text in enumerate(data) }
            if image_type == "Driving Licence":
                details = get_labels_from_licence(data)
            elif image_type == "Aadhar Card":
                details = get_labels_from_aadhar(data)

            return jsonify({'status':True, 'fields': details, 'image_path': filename, 'photo_path': photo_path})
    else:
        return jsonify({'status':False})


@app.route('/api/data', methods=['POST'])
def saveData():
    
    values = request.get_json()
    image_type = values.get('type')
    data = values.get('fields')
    
    db.insert_data(image_type, args_dict = data)

    return jsonify({'status': True})


@app.route('/image/face_match',methods=['GET','POST'])
def face_match():

    current_time = str(datetime.datetime.now())

    if not os.path.exists(UPLOAD_FOLDER + 'temp'):
            os.mkdir(UPLOAD_FOLDER + 'temp')

    filename = UPLOAD_FOLDER + 'temp' + '/' + current_time + '.png'
    
    photo_path = request.form['photopath']

    photo = request.files['liveface']
    photo.save(filename)
    
    result, percent = match_faces(id_card_image=photo_path, ref_image=filename)

    os.remove(filename)
    return jsonify({'status':str(result), 'percent': percent})


# GET
@app.route('/test')
def test():

    return "Return Test"


# running web app in local machine
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
