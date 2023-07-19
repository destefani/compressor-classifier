import os
from pathlib import Path
import requests
import tarfile
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle

from config import config

class ACLIMDB:
    def __init__(self, root_dir=Path(config.BASE_DIR, 'datasets'), shuffle=False, download=True):
        self.root_dir = root_dir
        self.dataset_dir = os.path.join(self.root_dir, 'aclImdb')
        self.shuffle = shuffle
        self.data = {
            "train": {"pos": [], "neg": []},
            "test": {"pos": [], "neg": []},
        }

        if download:
            os.makedirs(self.root_dir)
            self.download_data()

    def download_data(self):
        url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"  # Ensure this url is correct
        response = requests.get(url, stream=True)

        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kbyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        tar_gz_path = os.path.join(self.root_dir, "aclImdb_v1.tar.gz")
        with open(tar_gz_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        
        self.unzip_data(tar_gz_path)

    def unzip_data(self, tar_gz_path):
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            tar.extractall(path=self.root_dir)


    def load_data(self):
        for dataset_type in ["train", "test"]:
            for sentiment in ["pos", "neg"]:
                folder = os.path.join(self.dataset_dir, dataset_type, sentiment)
                print(folder)
                for filename in os.listdir(folder):
                    with open(os.path.join(folder, filename), 'r') as file:
                        text = file.read()
                        self.data[dataset_type][sentiment].append(text)
        print("Data loading complete!")

    def get_dataframe(self, dataset_type="train"):
        pos_data = self.data[dataset_type]["pos"]
        neg_data = self.data[dataset_type]["neg"]
        all_data = pos_data + neg_data
        labels = [1] * len(pos_data) + [0] * len(neg_data)

        df = pd.DataFrame({"text": all_data, "label": labels})

        # Shuffle the dataframe if the shuffle attribute is True
        if self.shuffle:
            df = shuffle(df, random_state=42)

        return df

    def get_train_test_data(self, test_size=0.2, random_state=42):
        train_df = self.get_dataframe("train")
        test_df = self.get_dataframe("test")
        
        X_train, y_train = train_df["text"].values, train_df["label"].values
        X_test, y_test = test_df["text"].values, test_df["label"].values
        
        return X_train, X_test, y_train, y_test
