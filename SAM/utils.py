import os
import sys
import requests
import base64
from tqdm import tqdm
from huggingface_hub import hf_hub_download

MODEL_URLS = {
    'sam_vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'sam_vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'sam_vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
}


def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 1))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    else:
        print("Something went wrong while downloading models")
        sys.exit(0)


def check_prefix(model_type, model_dir):
    model_path = ''
    files = os.listdir(model_dir)

    for filename in files:
        if filename.startswith(model_type) and filename.endswith(".pth"):
            model_path = os.path.join(model_dir, filename)

    return model_path


def load_model_hf_grounddino(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    return cache_config_file, cache_file


def check_weights(model_type, model_dir, save_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_path = check_prefix(model_type, model_dir)

    if model_path == '':
        url = MODEL_URLS[model_type]
        save_path = os.path.join(save_dir, url.split('/')[-1])
        download_with_progressbar(url, save_path)
        model_path = check_prefix(model_type, save_dir)
    return model_path


def bytes_to_base64(body):
    base64_encoded_data = base64.b64encode(body)
    base64_message = base64_encoded_data.decode('utf-8')
    return base64_message


def base64_to_bytes(base64_message):
    base64_img_bytes = base64_message.encode('utf-8')
    decoded_image_data = base64.decodebytes(base64_img_bytes)
    return decoded_image_data
