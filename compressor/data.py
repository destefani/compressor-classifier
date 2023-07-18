import os
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle

from config import config

class ACLIMDB:
    def __init__(self, root_dir=Path(config.BASE_DIR, 'datasets/aclimdb'), shuffle=False):
        self.root_dir = root_dir
        self.shuffle = shuffle
        self.data = {
            "train": {"pos": [], "neg": []},
            "test": {"pos": [], "neg": []},
        }

    def load_data(self):
        for dataset_type in ["train", "test"]:
            for sentiment in ["pos", "neg"]:
                folder = os.path.join(self.root_dir, dataset_type, sentiment)
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
