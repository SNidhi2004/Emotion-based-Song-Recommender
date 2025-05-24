# Training the model

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import joblib

# Load data
def load_data(path):
    texts, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ';' in line:
                text, label = line.strip().split(';')
                texts.append(text)
                labels.append(label)
    return pd.DataFrame({'text': texts, 'label': labels})

label_encoder = LabelEncoder()

train_df = load_data('data/train.txt')
test_df = load_data('data/test.txt')

# Label encode targets

train_df['labels'] = label_encoder.fit_transform(train_df['label'])
test_df['labels'] = label_encoder.transform(test_df['label'])

# Tokenize the text into numbers 

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_encoder.classes_))

# Trainer
training_args = TrainingArguments(
    output_dir="./model_output",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Train
trainer.train()

eval_results = trainer.evaluate()
print("\n📊 Evaluation Results:")
print(eval_results)

# Save model and label encoder
model.save_pretrained("model/distilbert")
tokenizer.save_pretrained("model/distilbert")
joblib.dump(label_encoder, "model/label_encoder.pkl")
