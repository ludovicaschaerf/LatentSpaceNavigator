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
import PIL

app = Flask(__name__)
CORS(app)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model_file = 'data/network-snapshot-005000.pkl'
pca_file = 'data/pca.pkl'

with dnnlib.util.open_url(model_file) as f:
     model = legacy.load_network_pkl(f)['G_ema'] # type: ignore

pca_reloaded = pickle.load(open(pca_file,'rb')) 
start_vec = np.array([-1.94338412, -1.22391922,  0.32755781])
# Define the root route
@app.route('/')
def home():
    return 'Welcome to the Flask App!'

def convert_position(position):
    position = np.array(position) #+ start_vec
    position_512 = pca_reloaded.inverse_transform(np.array(position).reshape(1,-1))
    print(position_512.shape)
    return position_512

def generate_image(vec):
    G = model.to(device) # type: ignore
    label = torch.zeros([1, G.c_dim], device=device)
    W = torch.from_numpy(np.repeat(vec, 16, axis=0).reshape(1, 16, vec.shape[1]).copy()).to(device)
    img = G.synthesis(W, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')    
            
        
def get_image_as_base64(position):
    print('position', position)
    vec = convert_position(position)
    pil_img = generate_image(vec)
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
    return encoded_img

@app.route('/get-image', methods=['POST'])
def send_image():
    position = request.json
    encoded_img = get_image_as_base64(position)
    return jsonify(imageData=encoded_img)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
