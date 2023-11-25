import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from scipy.special import softmax
import os

def preper_data(data_url):
    df = pd.read_csv(data_url)
    df_shuffled = df.sample(frac=1, random_state=42)

    train_ratio = 0.8
    train_size = int(train_ratio * len(df_shuffled))
    train_df = df_shuffled.head(train_size)
    test_df = df_shuffled.tail(len(df_shuffled) - train_size)

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, test_df

def plot_confusion_matrix(true_labels, predictions, title):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def fine_tune_and_evaluate(experiment_name, train_dataset, test_dataset, num_folds=3):
    model_type = 'bert'
    model_case = 'bert-base-uncased'
    model_directory = f"outputs/{experiment_name}"        
    model_args = ClassificationArgs(num_train_epochs=3,
                                    overwrite_output_dir=True,
                                    output_dir=model_directory,
                                    save_best_model=True,
                                    reprocess_input_data=True,
                                    no_cache=True,
                                    max_seq_length=16)

    model = ClassificationModel(model_type, model_case, args=model_args, use_cuda=False, num_labels=4)

    if num_folds > 1:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset, train_dataset['labels'])):
            train_fold = train_dataset.iloc[train_idx]
            val_fold = train_dataset.iloc[val_idx]
            model.train_model(train_fold)
    else:
        model.train_model(train_dataset)

    # Evaluation
    result, model_outputs, wrong_predictions = model.eval_model(test_dataset)
    predictions = model_outputs.argmax(axis=1)

    return predictions, test_dataset['labels']


def load_best_model(model_type='bert', model_case='bert-base-uncased', best_model_directory="outputs/bert_room_classification"):
    """
    Loads the best saved model from the specified directory.

    Args:
    model_type (str): The type of model (e.g., 'bert', 'distilbert').
    model_case (str): The specific case or version of the model (e.g., 'bert-base-uncased').
    best_model_directory (str): Path to the directory containing the best model.

    Returns:
    model: The loaded ClassificationModel.
    """
    model_args = ClassificationArgs()    
    model = ClassificationModel(model_type, best_model_directory, args=model_args,use_cuda=False)
    
    return model

if __name__ == '__main__':
    pass
    # Train and evaluate the model Bert in-domain
    # data_url = 'https://raw.githubusercontent.com/Mohammed-majeed/Communicative-Robots/main/processed_room_classification_data.csv'
    # train_df, test_df = preper_data(data_url)
    # predictions, true_labels = fine_tune_and_evaluate("bert_room_classification", train_df, test_df, num_folds=3)
    # plot_confusion_matrix(true_labels, predictions, "Confusion Matrix for BERT")
    
    # best_model_directory = "outputs/bert_room_classification"
    # best_model = load_best_model('bert', 'bert-base-uncased', best_model_directory)
    # x = best_model.predict(['test'])
    # print(x)
