from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import Train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
bert_model = distilbert_model.to(device)

Train.Train_model(bert_model,tokenizer,'esgn_dataset.xlsx')