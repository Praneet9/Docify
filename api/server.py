from flask import Flask, request, jsonify
import os
from processing import recognise_text, crop_aadhar, get_address, seven_segment, _init_model, get_labels_from_aadhar, get_labels_from_licence
from cheque_details_extraction import get_micrcode, ensemble_acc_output, ensemble_ifsc_output
import datetime
import db
from face_matching import match_faces

app = Flask(__name__)

# path to upload images
UPLOAD_FOLDER = './UPLOAD_FOLDER/'

# initializing seven segment display model
_init_model()

# route to uploading images of id cards
@app.route('/image/upload', methods=['POST'])
def index():
    
    if request.method == 'POST':

        # saving current timestamp
        current_time = str(datetime.datetime.now())

        # get the type of image that is being received
        image_type = request.form['type']
        
        # setting filename that is being received to current time stamp with its directory
        filename = UPLOAD_FOLDER + image_type + '/' + current_time + '.png'

        # if the image_type folder doesn't already exist, create it
        if not os.path.exists(UPLOAD_FOLDER + image_type):
            os.mkdir(UPLOAD_FOLDER + image_type)
            # directory for saving faces in the id cards
            os.mkdir(UPLOAD_FOLDER + image_type + '/' + 'faces')
        
        # if image_type is bank cheque, preprocess accordingly
        if image_type == 'Bank Cheque':
            details = {}

            # get photo from android
            photo = request.files['photo']
            photo.save(filename)

            # get details from the image
            details['MICR'] = get_micrcode(filename)
            details['ACC.No'] = ensemble_acc_output(filename)
            details['IFSC'] = ensemble_ifsc_output(filename)

            # return the details and the image name it is saved as
            return jsonify({'status':True, 'fields': details, 'image_path': filename, 'photo_path': 'none' })

        # if image_type is seven segment, preprocess accordingly
        elif image_type == 'Seven Segment':
            details = {}

            # get photo from android
            photo = request.files['photo']
            photo.save(filename)

            # get text from seven segment
            text = seven_segment(filename)

            details[0] = text

            # return the details and the image name it is saved as
            return jsonify({'status':True, 'fields': details, 'image_path': filename, 'photo_path': 'none' })

        # elif image_type == 'Aadhar Back':
        #     details = {}

        #     # get photo from android
        #     photo = request.files['photo']
        #     photo.save(filename)

        #     crop_path = UPLOAD_FOLDER + image_type + '/temp/' + current_time + '.png'

        #     if not os.path.exists(UPLOAD_FOLDER + image_type + '/temp'):
        #         os.mkdir(UPLOAD_FOLDER + image_type + '/temp')

        #     crop_aadhar(filename, crop_path)

        #     # recognise text in the id card
        #     data, photo_path = recognise_text(crop_path, 'none')
            
        #     details = get_address(data)

        #     os.remove(crop_path)

        #     # return the details and the image name it is saved as
        #     return jsonify({'status':True, 'fields': details, 'image_path': filename, 'photo_path': 'none' })
        
        else:
            # setting directory for saving face in the id card
            photo_path = UPLOAD_FOLDER + image_type + '/' + 'faces' + '/' + current_time + '.png'
            
            # get photo from android
            photo = request.files['photo']
            photo.save(filename)

            # recognise text in the id card
            data, photo_path = recognise_text(filename, photo_path)
            
            # extract labels from the recognised text according to the image_type
            if image_type == "Driving Licence":
                details = { idx : text for idx, text in enumerate(data) }
                details = get_labels_from_licence(details)
            elif image_type == "Aadhar Card":
                details = get_labels_from_aadhar(data)
            else:
                details = { idx : text for idx, text in enumerate(data) }

            with open('outputs.txt', 'a+') as f:
                f.write("##########################################################################\n\n")
                f.write('######################## Raw Output #############################\n\n')
                for value in data:
                    f.write(str(value) + '\n')
                f.write('\n\n######################## Cleaned Output #############################\n\n')
                for key, value in details.items():
                    f.write(str(key) + ' : ' + str(value) + '\n')
                f.write("##########################################################################\n\n")

            # return the details and the image name and photo path it is saved as
            return jsonify({'status':True, 'fields': details, 'image_path': filename, 'photo_path': photo_path})
    else:
        # if not POST, terminate
        return jsonify({'status':False})

# save data to database
@app.route('/api/data', methods=['POST'])
def saveData():
    
    # get values as json
    values = request.get_json()
    image_type = values.get('type')
    data = values.get('fields')
    
    db.insert_data(image_type, args_dict = data)

    return jsonify({'status': True})


@app.route('/image/face_match',methods=['GET','POST'])
def face_match():

    # saving current timestamp
    current_time = str(datetime.datetime.now())

    # temporary folder for saving face for face matching
    if not os.path.exists(UPLOAD_FOLDER + 'temp'):
            os.mkdir(UPLOAD_FOLDER + 'temp')

    # setting filename that is being received to current time stamp with its directory
    filename = UPLOAD_FOLDER + 'temp' + '/' + current_time + '.png'
    
    # getting the path of the saved face image
    photo_path = request.form['photopath']

    # get live face from android
    photo = request.files['liveface']
    photo.save(filename)
    
    # check face match and probability
    result, percent = match_faces(id_card_image=photo_path, ref_image=filename)

    # delete the temp face image
    os.remove(filename)

    # return face match prediction and percentage
    return jsonify({'status':str(result), 'percent': percent})


# GET
@app.route('/test')
def test():

    return "Return Test"


# running web app in local machine
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
