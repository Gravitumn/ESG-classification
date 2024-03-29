import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout , SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import BertTokenizer, BertForSequenceClassification,BertModel
from transformers import BertPreTrainedModel
import matplotlib.pyplot as plt
import spacy
import re
import string
import seaborn as sns
from keras import Model
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import Layer
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim import AdamW
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def cleaning(path,test_size, random_state):
    data = pd.read_excel(path)
    ori_text = data['ESGN'].values

    # Removing Punctuations
    data['ESGN'] = data['ESGN'].apply(lambda x: x.translate(str.maketrans('','', string.punctuation)))

    # Removing urls
    data['ESGN']=data['ESGN'].apply(lambda x : re.compile(r'https?://\S+|www\.\S+').sub('',x))

    # Removing HTML Tags
    data['ESGN']=data['ESGN'].apply(lambda x : re.compile(r'<.*?>').sub('',x))
    
    # Removing emoji tags
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    data['ESGN']=data['ESGN'].apply(lambda x : emoji_pattern.sub('',x))

    # lowercase
    data['ESGN']=data['ESGN'].apply(lambda x : x.lower())

    # Stop word
    nlp = spacy.load("en_core_web_sm")
    def stop_word(text):
        temp=[]
        for t in nlp(text):
            if not nlp.vocab[t.text].is_stop :
                temp.append(t.text)
        return " ".join(temp)

    data['ESGN']=data['ESGN'].apply(lambda x : stop_word(x) )
    
    # Split data into train and validation sets
    sentences = data['ESGN'].values
    labels = data['class'].values
    
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state= random_state)
    train_indices, val_indices = next(stratified_splitter.split(sentences, labels, ori_text))
    
    # Split the data based on the indices obtained from stratified sampling
    train_text = [sentences[i] for i in train_indices]
    val_text = [sentences[i] for i in val_indices]
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    train_ori_text = [ori_text[i] for i in train_indices]
    val_ori_text = [ori_text[i] for i in val_indices]
    
    return train_text,val_text,train_labels,val_labels,train_ori_text,val_ori_text



def Train_model(bert_model,tokenizer,device, file_path,batch_size = 8,
                shuffle = True, lr = 1e-5, num_epochs = 100,
                T_0 = 10, T_mult=2, eta_min = 1e-6,
                model_save_path = f'/model/trained_model', early_stop_epoch = 8, return_result = False, test_size = 0.1,random_state = 69):
    train_text,val_text,train_labels,val_labels,train_ori_text,val_ori_text = cleaning(file_path,test_size, random_state)
    best_model = None
    
    # tokenize the text data
    train_list = [str(t) for t in train_text]
    val_list = [str(t) for t in val_text]
    train_inputs = tokenizer(train_list, padding=True, truncation=True, return_tensors="pt")
    val_inputs = tokenizer(val_list, padding=True, truncation=True, return_tensors="pt")

    # convert labels to PyTorch tensors
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)

    # create DataLoader for training and validation sets
    train_dataset = TensorDataset(train_inputs.input_ids, train_inputs.attention_mask, train_labels)
    val_dataset = TensorDataset(val_inputs.input_ids, val_inputs.attention_mask, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = AdamW(bert_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for param in bert_model.parameters():
        param = param.to(device)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []
    best_accuracy = 0
    best_epoch = 0
    early_stop = False
    
    # Training loop
    for epoch in range(num_epochs):
        bert_model.train()
        train_predictions = []
        train_true_labels = []
        train_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            labels = labels.to(torch.long)
            loss = loss_fn(probabilities, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            predicted_class = torch.argmax(probabilities, dim=-1)
            train_predictions.extend(predicted_class.cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(train_true_labels, train_predictions)
        train_accuracies.append(train_accuracy * 100)
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Train Loss: {avg_train_loss}")
        print(f"Train Accuracy: {train_accuracy}")

        # Validation
        bert_model.eval()
        test_predictions = []
        test_true_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = bert_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                labels = labels.to(torch.long)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()

                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1)
                test_predictions.extend(predicted_class.cpu().numpy())
                test_true_labels.extend(labels.cpu().numpy())

            val_accuracy = accuracy_score(test_true_labels, test_predictions)
            val_accuracies.append(val_accuracy * 100)
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss}")
            print(f"Validation Accuracy: {val_accuracy * 100}")

            # Update the scheduler
            scheduler.step()

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                # Save the model
                bert_model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                best_model = bert_model
            else:
                # Check for early stopping
                if epoch - best_epoch >= early_stop_epoch:
                    early_stop = True
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
        if return_result:
            test_loss=0
            test_predictions = []
            test_true_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch

                    # Move input data to GPU
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)

                    outputs = best_model(input_ids, attention_mask=attention_mask)
                    outputs = outputs[0]

                    labels = labels.to(torch.long)
                    loss = loss_fn(outputs, labels)
                    test_loss += loss.item()
                    # Apply softmax to obtain probabilities
                    probabilities = torch.nn.functional.softmax(outputs, dim=-1)

                    # Get predicted class (index with the highest probability)
                    predicted_class = torch.argmax(probabilities, dim=-1)

                    test_predictions.extend(predicted_class.cpu().numpy())
                    test_true_labels.extend(labels.cpu().numpy())

                test_accuracy = accuracy_score(test_true_labels, test_predictions)

                # Calculate average validation loss for the epoch
                avg_test_loss = test_loss / len(val_loader)
                
            # Plotting the training and validation losses
            plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Training accuracy')
            plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.show()
            
            # Compute the confusion matrix
            conf_matrix = confusion_matrix(test_true_labels, test_predictions)

            # Plot the confusion matrix
            class_names = ['e','s','g','none']
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
            
            # Calculate precision, recall, and F1 score
            precision = precision_score(test_true_labels, test_predictions, average='weighted')
            recall = recall_score(test_true_labels, test_predictions, average='weighted')
            f1 = f1_score(test_true_labels, test_predictions, average='weighted')
            
            print(f"Test Loss: {test_loss / len(val_loader)}")
            print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
            print(f"Precision: {precision * 100:.2f}%")
            print(f"Recall: {recall * 100:.2f}%")
            print(f"F1 Score: {f1 * 100:.2f}%")
            
            results_df = pd.DataFrame({
                  'Original Text': val_ori_text,
                  'Predicted Label': test_predictions,
                  'True Label': test_true_labels
            })

            # Save the DataFrame to a CSV file
            results_df.to_csv('prediction_result.csv', index=False)

            # Calculate precision, recall, and F1 score
            precision = precision_score(test_true_labels, test_predictions, average='weighted')
            recall = recall_score(test_true_labels, test_predictions, average='weighted')
            f1 = f1_score(test_true_labels, test_predictions, average='weighted')
            
            if not early_stop:
                best_model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                