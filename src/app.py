from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import legacy 
import dnnlib
import pickle
import torch
import numpy as np
import json
import PIL

app = Flask(__name__)
CORS(app)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model_file = 'data/network-snapshot-005000.pkl'
pca_file = 'data/pca.pkl'
directions_3d_file = 'public/3d_directions.json'
directions_512d_file = 'public/512d_directions.json'

with dnnlib.util.open_url(model_file) as f:
     model = legacy.load_network_pkl(f)['G_ema'] # type: ignore

pca_reloaded = pickle.load(open(pca_file,'rb')) 
start_vec = np.array([-1.94338412, -1.22391922,  0.32755781])

with open(directions_512d_file, "r") as infile: 
    directions_512d = json.load(infile)
with open(directions_3d_file, "r") as infile: 
    directions_3d = json.load(infile)

# Define the root route
@app.route('/')
def home():
    return 'Welcome to the Flask App!'

def convert_position(color, oldpos):
    color512d = directions_512d[color]
    position_512 = np.array(oldpos) + np.array(color512d)
    #position_512 = pca_reloaded.inverse_transform(np.array(position).reshape(1,-1))
    return position_512

def generate_image(vec):
    G = model.to(device) # type: ignore
    label = torch.zeros([1, G.c_dim], device=device)
    W = torch.from_numpy(np.repeat(vec, 16, axis=0).reshape(1, 16, 512).copy()).to(device)
    img = G.synthesis(W, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')    
            
        
def get_image_as_base64(color, oldpos):
    if color != '':
        vec = convert_position(color, oldpos).reshape((1,512))
    else:
        vec = np.array(oldpos).reshape((1,512))
    newPosition = [float(v) for v in list(vec.reshape(512))]
    pil_img = generate_image(vec)
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
    return encoded_img, newPosition

@app.route('/get-image', methods=['POST'])
def send_image():
    input_data = request.json
    color, oldpos = input_data
    encoded_img, newPosition = get_image_as_base64(color, oldpos)
    return jsonify(imageData=encoded_img, newPosition=newPosition)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
