# Load model directly
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from tqdm import tqdm
import torch.nn as nn
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]["Window"]
        label = self.data[index]["TruthValue"]
        encoded_inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=16,  # Adjust the maximum sequence length as needed
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()

        return input_ids, attention_mask, label

# Load CSV file
csv_file_path = "train.csv"  # Update this path to your CSV file
df = pd.read_csv(csv_file_path)

# Convert DataFrame to a list of dictionaries
train_data = df.to_dict('records')

# Initialize CustomDataset with the data
train_dataset = CustomDataset(train_data)


tokenizer = AutoTokenizer.from_pretrained("unikei/bert-base-proteins")
model = AutoModelForSequenceClassification.from_pretrained("unikei/bert-base-proteins")

device = torch.device('cuda')
model.to(device)

optimizer = AdamW(model.parameters(), lr=4e-5)  # Adjust the learning rate as needed

train_dataloader = DataLoader(train_dataset, batch_size=32)

num_epochs = 5  # Adjust the number of epochs as needed

dropout_prob = 0.001  # You can adjust this value based on experimentation
model.classifier = nn.Sequential(
    nn.Dropout(dropout_prob),
    nn.Linear(model.config.hidden_size, 2)
).to(device)
class_weights = torch.tensor([5.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in tqdm(train_dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_ids = labels.clone().detach().to(device) # Convert labels to integers using the label mapping
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=label_ids)
        loss = criterion(outputs.logits, label_ids)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {average_loss}")


# Step 1: Load the test CSV file
test_df = pd.read_csv("test.csv")

# Step 2: Convert DataFrame to a list of dictionaries
test_data = test_df.to_dict('records')

# Initialize CustomDataset with the test data
test_dataset = CustomDataset(test_data)

# Create a DataLoader for the test data
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 3: Loop through the DataLoader to get model predictions
model.eval()  # Put the model in evaluation mode
predictions = []
true_labels = []

with torch.no_grad():  # Disable gradient calculation
    for input_ids, attention_mask, labels in tqdm(test_dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.numpy())

# Optionally, calculate metrics like accuracy here if labels are available
# For example:
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy}")
