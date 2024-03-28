import re
import nltk
from nltk.tokenize import sent_tokenize
import fitz
import os
import csv
import spacy
import numpy as np
import math
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_sm")

import pandas as pd
import torch
from transformers import AlbertForSequenceClassification, AlbertTokenizer

# Load ALBERT model and tokenizer
model_name = f'model/albert2'  # You can change this to a different ALBERT variant if needed
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def remove_extra_whitespace(sentences):
    # Removing extra spaces and tabs from each sentence in the list
    return [re.sub(r'\s+', ' ', sentence).strip() for sentence in sentences]

def remove_URLs(texts):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return [url_pattern.sub(r'', text) for text in texts]

def remove_special_chars(sentences):
  cleaned_sentences = []
  for sentence in sentences:
    cleaned_sentences.append(sentence.replace("", "").replace("*","").replace("","").replace("$","").replace("#","").replace("+","").replace("|",""))
  return cleaned_sentences

def extract_text_from_pdf(file_path):
    report_text = ""
    with fitz.open(file_path) as pdf_document:
        num_pages = pdf_document.page_count
        for page_num in range(num_pages):
            page = pdf_document[page_num]
            report_text += page.get_text("text")
    return report_text

def tokenize_sentences(text):
    return sent_tokenize(text)

def remove_table_of_contents(sentences):
    cleaned_sentences = []
    count = 0
    for sentence in sentences:
        # Removing leading and trailing whitespaces
        sentence = sentence.strip()
        sentence = re.sub(r'\s+', ' ', sentence)

        # Ignore sentences starting with "Table of Contents" or "Contents"
        if (
            sentence.lower().startswith("table of contents")
            or sentence.lower().startswith("contents")
            or re.search(r'\.{6,}', sentence)  # Check for multiple dots
            or re.search(r'-\s*-\s*-', sentence)
        ):
            count+=1
            global removed_sentence
            removed_sentence.append(sentence)
            continue
        else:
            cleaned_sentences.append(sentence)
    return cleaned_sentences,count

def remove_long_short_sentence(sentences):
  cleaned_sentences = []
  count = 0
  for sentence in sentences:
    if(
        len(sentence.split()) <=5
#         or len(sentence.split()) >= 50
    ):
      count+=1
      global removed_sentence
      removed_sentence.append(sentence)
      continue
    else:
      cleaned_sentences.append(sentence)
  return cleaned_sentences,count

def remove_head(sentences):
    cleaned_sentences = []

    for sentence in sentences:
        # Split the sentence into words
        words = sentence.split()

        # Initialize variables to store the start and end positions
        start_index = None
        end_index = None

        # Find the start and end positions
        for i, word in enumerate(words):
            if word.isupper() and word != "I":
                if start_index is None:
                    start_index = i
                end_index = i

        # Remove words between the start and end positions
        if start_index is not None and end_index is not None:
            cleaned_sentence = ' '.join(words[:start_index] + words[end_index + 1:])
            cleaned_sentences.append(cleaned_sentence)
        else:
            cleaned_sentences.append(sentence)

    return cleaned_sentences

def help_ie(input_string):
    index_ie = input_string.find("i.e.")
    while index_ie != -1:
        # Add a comma after "i.e."
        input_string = input_string[:index_ie + 4] + "," + input_string[index_ie + 4:]
        # Look for the next occurrence of "i.e."
        index_ie = input_string.find("i.e.", index_ie + 1)
    return input_string

def split_number_bullet(sentences):
    cleaned_sentences = []
    
    for sentence in sentences:
        words = sentence.split()
        extracted_words = []
        for word in words:
            match = re.search(r'\(?\d\)', word)
            if match:
                extracted_words.extend([word[:match.start()], match.group(), word[match.end():]])
            else:
                extracted_words.append(word)
        indices = []
        for i, word in enumerate(extracted_words):
            match = re.search(r'\(?\d\)', word)
            if match:
                indices.append(i)
        distances = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        
        if len(indices) != 0:
            if all(distance > 5 for distance in distances):
                splitted_sentences = []
                start_index = 0
                for index in indices:
                    splitted_sentences.append(' '.join(extracted_words[start_index:index]))
                    start_index = index
                splitted_sentences.append(' '.join(extracted_words[start_index:index]))

                for w in splitted_sentences:
                    cleaned_sentences.append(w)
            else:
                cleaned_sentences.append(sentence)
        else:
            cleaned_sentences.append(sentence)
        cleaned_sentences = [s for s in cleaned_sentences if s.strip()]
    return cleaned_sentences

def split_bullet(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        split_sentence = sentence.split('•')
        cleaned_sentence = [s.strip() for s in split_sentence if s.strip()]
        cleaned_sentences.extend(cleaned_sentence)
    return cleaned_sentences

def calculate_digit_percentage(sentence):
    # Helper function to calculate the percentage of digits in a string
    total_chars = len(sentence)
    digit_chars = sum(char.isdigit() for char in sentence)
    return (digit_chars / total_chars) * 100

def remove_too_much_digit(sentences):
  cleaned_sentences= []
  count=0
  for sentence in sentences:
    if calculate_digit_percentage(sentence) >= 30:
      count+=1
      global removed_sentence
      removed_sentence.append(sentence)
      continue
    else:
      cleaned_sentences.append(sentence)
  return cleaned_sentences,count

def is_full_sentence(sentence):
    # Parse the sentence
    doc = nlp(sentence)
    
    # Check if the sentence contains a subject and a predicate (verb phrase)
    has_subject = False
    has_predicate = False
    for token in doc:
        # Check for subjects (nsubj and nsubjpass dependencies)
        if token.dep_ in ['nsubj', 'nsubjpass','nsubjpass','csubj','csubjpass','nsubjrel','nsubjcaus']:
            has_subject = True
        # Check for verbs and auxiliaries (root and aux dependencies)
        elif token.dep_ in ['ROOT', 'aux']:
            has_predicate = True
        elif token.lower_ == 'there' and token.i < len(doc) - 1 and doc[token.i + 1].lemma_ == 'be':
            has_subject = True
            has_predicate = True
    # Return True if both subject and predicate are present
    return has_subject and has_predicate

def remove_phrase(sentences):
  sent  = []
  count = 0
  for sentence in sentences:
    if is_full_sentence(sentence):
      sent.append(sentence)
    else:
      global removed_sentence
      removed_sentence.append(sentence)
      count+=1
  return sent,count

def pdf_to_csv(company,start_year,end_year,pdf_folder):
    year = start_year
    end_year = end_year
    global removed_sentence
    removed_sentence = []
    for comp in company:
      print('current company :',comp)
      for i in range(year,end_year+1):
        removed_sentence = []
        print('year:',i)
        file_path_upper = f'{pdf_folder}{comp}_{i}.PDF'
        file_path_lower = f'{pdf_folder}{comp}_{i}.pdf'
        report_text = None
        if os.path.exists(file_path_upper):
            file_path = file_path_upper
        elif os.path.exists(file_path_lower):
            file_path = file_path_lower
        else:
            print(f"{comp}_{i}.PDF not found")
            continue
        try:
            report_text = extract_text_from_pdf(file_path=file_path)
        except FileNotFoundError:
            print(f"Error reading {file_path}")
            continue
        
        if report_text is None:
            print(f"Error reading {file_path}")
            continue
                
        try:
            report_text = help_ie(report_text)
            sents = tokenize_sentences(report_text)
            cleaned_sentences, table_content_count = remove_table_of_contents(sents)
            cleaned_sentences = remove_head(cleaned_sentences)
            cleaned_sentences = split_bullet(cleaned_sentences)
            cleaned_sentences = split_number_bullet(cleaned_sentences)
            cleaned_sentences, long_short_count = remove_long_short_sentence(cleaned_sentences)
            cleaned_sentences, phrase_count = remove_phrase(cleaned_sentences)
            cleaned_sentences, too_long_digit_count = remove_too_much_digit(cleaned_sentences)
            cleaned_sentences = remove_extra_whitespace(cleaned_sentences)
            cleaned_sentences = remove_URLs(cleaned_sentences)
            cleaned_sentences = remove_special_chars(cleaned_sentences)
            after_clean = len(cleaned_sentences)

            print("sentence amount =", len(sents))
            print("Is table of content:", table_content_count)
            print("Is too long or short:", long_short_count)
            print("Is phrase:", phrase_count)
            print("Is too much digit:", too_long_digit_count)
            print("Total of sentence after clean:", after_clean)

            output_file_path = os.path.join('', f"data_pdf/{comp}_{i}_sentences_output.csv")
            with open(output_file_path, "w", newline="", encoding="utf-8") as output_file:
                csv_writer = csv.writer(output_file)
                csv_writer.writerow(["Sentence Index", "Sentence"])
                for index, sentence in enumerate(cleaned_sentences, start=1):
                    csv_writer.writerow([index, sentence])

            remove_output_file_path = os.path.join('', f"data_pdf/{comp}_{i}_sentences_removed.csv")
            with open(remove_output_file_path, "w", newline="", encoding="utf-8") as output_file:
                csv_writer = csv.writer(output_file, escapechar='\\')
                csv_writer.writerow(["Sentence Index", "Sentence"])
                for index, sentence in enumerate(removed_sentence, start=1):
                    csv_writer.writerow([index, sentence])
        except Exception as e:
            if "Error: need to escape, but no escapechar set" in str(e):
                print("There are special characters that are not English")
            else:
                print(f"Error occurred while processing {file_path}: {e}")
            continue


def predict_pdf(csv_files,start,end):
    classes = ["E", "S", "G", 'N']
    company_counts = {}
    for csv_file in csv_files:
        company_counts[csv_file] = {}  # Initialize counts for the current company
        for i in range(start, end + 1):
            prediction_results = []
            # Extract company name from the filename
            company_name = csv_file  # Assuming the filename format is "company_name.csv"
            company_counts[csv_file][i] = {cls: 0 for cls in classes}  # Initialize counts for the current year
            
            try:
                # Load CSV data
                data = pd.read_csv(f'data_pdf/{csv_file}_{i}_sentences_output.csv')
            except FileNotFoundError:
                print(f"CSV file {csv_file} not found. Skipping...")
                continue

            # Iterate over sentences
            for sentence in data['Sentence']:
                # Tokenize the sentence
                inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
                inputs.to(device)

                # Perform inference
                with torch.no_grad():
                    outputs = model(**inputs)

                # Get predicted class index
                predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
                predicted_class = classes[predicted_class_idx]

                prediction_results.append(
                    {'Sentence': sentence, 'Predicted Class': predicted_class, 'Source CSV': csv_file})

                # Update counts for the current year
                company_counts[csv_file][i][predicted_class] += 1

            # Convert prediction results to DataFrame
            prediction_df = pd.DataFrame(prediction_results)

            # Save prediction results to CSV
            prediction_df.to_csv(f'output/prediction_results_{csv_file}_{i}.csv', index=False)
            print(f"Prediction results for {csv_file} - {i} saved to prediction_results.csv")

    
    for company, year_counts in company_counts.items():
        # Create a new figure for each company
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set width of bars
        bar_width = 0.2

        # Set position of bar on X axis
        r = range(len(year_counts))
        bar_positions = [i + bar_width * j for i in r for j in range(len(classes))]
        
        text_settings = dict(ha='center', va="bottom", fontsize=6)

        # Plot bars for each class
        for i, cls in enumerate(classes):
            # Initialize class counts for the current year
            class_counts_year = [year_counts[year][cls] for year in year_counts]
            ax.bar(bar_positions[i::len(classes)], class_counts_year, width=bar_width, label=cls)

            # Calculate the percentage of each class for each year
            total_count_year = [sum(year_counts[year][c] for c in classes) for year in year_counts]
            class_percentage_year = [class_counts_year[j] / total_count_year[j] * 100 if total_count_year[j] != 0 else 0 for j in range(len(class_counts_year))]

            # Add percentage text on top of bars
            for pos, percentage in zip(bar_positions[i::len(classes)], class_percentage_year):
                ax.text(pos, class_counts_year[int(pos / len(classes))] + 0.1 * max(class_counts_year), f'{percentage:.2f}%', **text_settings)

        # Add labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        ax.set_title(f'Count of each class for {company}')
        ax.set_xticks([r + bar_width for r in range(len(year_counts))])
        ax.set_xticklabels(list(year_counts.keys()))

        # Add legend
        ax.legend()

        # Show plot for each company
        plt.tight_layout()
        plt.savefig(f'graph/{company}.png')
        plt.close()

if __name__ == "__main__":
    pdf_folder = ''
    company = ['ADVANC']
    start_year = 2018
    end_year = 2022
    pdf_to_csv(company,start_year,end_year,pdf_folder)
    predict_pdf(company,start_year,end_year)