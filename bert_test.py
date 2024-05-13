import torch
import pandas as pd
from transformers import BertTokenizer


def predict(text, model, tokenizer):
    encoding = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_ids = encoding['input_ids'].to('cuda:0')
    attention_mask = encoding['attention_mask'].to('cuda:0')

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, axis=1)

    return predicted_class.item()


model = torch.load('content/bert.pt')
model.eval()

tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny')

test_data_p = pd.read_csv('data/test.csv')
texts = list(test_data_p['text'])

predictions = [predict(t, model, tokenizer) for t in texts]
test_data_p['rating'] = predictions
test_data_p[['ID', 'rating']].to_csv('sub_bert.csv', index=False)
