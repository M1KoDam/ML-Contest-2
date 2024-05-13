import torch

model = torch.load('bert.pt')
model.eval()

preds = model.predict(vectors_sub)
df_sub['rating'] = preds
df_sub[['ID', 'rating']].to_csv('sub_bert_catboost.csv', index=False)
