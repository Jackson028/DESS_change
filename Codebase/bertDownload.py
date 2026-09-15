import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)

def download_bert_model(model_name='bert-base-uncased', save_path='./bert-base-uncased'):
    base_url = f"https://huggingface.co/{model_name}/resolve/main/"
    files_to_download = [
        'config.json',
        'pytorch_model.bin',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.txt'
    ]

    os.makedirs(save_path, exist_ok=True)

    for file in files_to_download:
        url = base_url + file
        filepath = os.path.join(save_path, file)
        print(f"Downloading {file}...")
        download_file(url, filepath)

    print(f"BERT model '{model_name}' has been downloaded to {save_path}")

if __name__ == "__main__":
    download_bert_model()