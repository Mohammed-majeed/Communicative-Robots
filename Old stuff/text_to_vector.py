from simpletransformers.classification import ClassificationModel, ClassificationArgs
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch



def encode_text_with_bert(text):
    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the text
    input_ids = tokenizer.encode(text, add_special_tokens=True)

    # Convert input IDs to PyTorch tensor
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    # Get BERT embeddings for the input text
    with torch.no_grad():
        outputs = model(input_ids)
        bert_embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling to get a fixed-size embedding vector

    return bert_embeddings