import pandas as pd
import random



def load_data(url='https://raw.githubusercontent.com/Mohammed-majeed/Communicative-Robots/main/dataset.csv'):
    df = pd.read_csv(url)
    # Convert object names to lowercase
    df['Object'] = df['Object'].str.lower()
    return df


def data_preprocessing(df=load_data()):
    # inreach the dataset
    labels = df['Label'].unique()
    concatenated_dfs = []

    for label in labels:
        label_objects = df[df['Label'] == label].copy()  
        temp = label_objects['Object'].tolist()
        random.shuffle(temp)
        label_objects.loc[:, 'Object_1'] = temp  
        random.shuffle(temp)
        label_objects.loc[:, 'Object_2'] = temp 
        concatenated_dfs.append(label_objects)

    # Concatenate all DataFrames
    result = pd.concat(concatenated_dfs, ignore_index=True)

    temp = result['Object'].tolist()
    random.shuffle(temp)
    result['Object_3'] = temp

    # Concatenate object columns into a single text column
    result['text'] = result['Object'] + ' ' + result['Object_1'] + ' ' + result['Object_2'] + ' ' + result['Object_3']

    df = data_encoder(result)[['text', 'labels']]
    return df

def data_encoder(result):
    # Encode the labels in the 'Label' column
    encode_label={'bedroom':0, 'living room':1, 'bathroom':2, 'kitchen':3}
    result['labels'] = result['Label'].map(encode_label)
    return result



if __name__ == '__main__':        
    processed_data = data_preprocessing()
    processed_data.to_csv('processed_room_classification_data.csv', index=False)
