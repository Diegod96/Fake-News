import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def clean_data(file_paths):
    """
    Cleans data from all three datasets and combines them into one dataset for the model
    Schema is title, text, label
    :param file_paths:
    :return: all_data
    """

    # Assign file path for each dataset from filepaths
    data_file_path = file_paths[0]
    fake_file_path = file_paths[1]
    true_file_path = file_paths[2]

    # Read in data.csv
    # Drop extra columns
    # Assign 1 as FAKE and 0 as TRUE
    # Rename column names
    data = pd.read_csv(data_file_path)
    data = data.drop(columns=['URLs'])
    data['Label'] = data['Label'].replace([1], 'FAKE')
    data['Label'] = data['Label'].replace([0], 'TRUE')
    data = data.rename(columns={"Headline": "title", 'Body': 'text', "Label": 'label'})

    # Read in Fake.csv
    # Drop extra columns
    # Add "label" column and give it FAKE value
    fake = pd.read_csv(fake_file_path)
    fake = fake.drop(columns=['date'])
    fake = fake.drop(columns=['subject'])
    fake['label'] = 'FAKE'

    # Read in True.csv
    # Drop extra columns
    # Add "label" column and give it TRUE value
    true = pd.read_csv(true_file_path)
    true = true.drop(columns=['date', 'subject'])
    true['label'] = 'TRUE'

    # Create all_data df
    # Append all previous dataframes to it
    # Shuffle the data
    all_data = data.append(fake)
    all_data = all_data.append(true)
    all_data = all_data.sample(frac=1).reset_index(drop=True)

    return all_data


if __name__ == '__main__':
    file_paths = ['data/data.csv', 'data/Fake.csv', 'data/True.csv']
    all_data = clean_data(file_paths)
