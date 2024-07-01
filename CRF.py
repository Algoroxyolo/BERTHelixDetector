from torchcrf import CRF
train_file_path = "IO.txt"
test_file="test-noi.txt"
def read_data(file_path):
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split('\n')
        for line in lines:
            if line:
                word, label = line.split('\t')
                current_sentence.append(word)
                current_labels.append(label)
            else:
                sentences.append(current_sentence)
                labels.append(current_labels)
                current_sentence = []
                current_labels = []
    return sentences,labels

sent,tags=read_data(train_file_path)
tr_sent,tr_tag=read_data(test_file)


def create_formatted_data(id, ner_tags, tokens):
    formatted_data = {
        'id': id,
        'ner_tags': ner_tags,
        'tokens': tokens
    }
    return formatted_data

i=0
XCOLD=[]
for token,tag in zip(sent,tags):
    XCOLD.append(create_formatted_data(i,tag,token))
from datasets import Dataset
XCOLD = Dataset.from_list(XCOLD)

XCOLD_test=[]
for token,tag in zip(tr_sent,tr_tag):
    XCOLD_test.append(create_formatted_data(i,tag,token))
XCOLD_test = Dataset.from_list(XCOLD_test)

from transformers import AutoTokenizer
import torch
import signal
'''def handler_function(a,b):
    print("in handler")
    print(torch.cuda.is_available())
signal.signal(signal.SIGALRM,handler_function)
signal.alarm(5)
print(torch.cuda.is_available())'''
# Define label-to-id and id-to-label mappings
id2label = {
    0: "B",
    1: "I",
    2: "O"
}

label2id = {
    "B": 0,
    "I": 1,
    "O": 2
}
# Load the tokenizer with id2label and label2id mappings
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", id2label=id2label, label2id=label2id)

tokenizer.bos_token = "<BOS>"
tokenizer.eos_token = "<EOS>"

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], padding="max_length", truncation=True, is_split_into_words=True, return_tensors="pt", max_length=64)
    label_id_sequences = [[label2id[label] for label in instance_labels] for instance_labels in examples["ner_tags"]]  # Convert labels to IDs
    labels = []
    for i, label_id_sequence in enumerate(label_id_sequences):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_sequence = []
        for word_idx in word_ids:
            if word_idx is None:
                label_sequence.append(-100)
            elif word_idx != previous_word_idx:
                label_sequence.append(label_id_sequence[word_idx])
            else:
                label_sequence.append(-100)
            previous_word_idx = word_idx
        labels.append(label_sequence)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


XCOLD = XCOLD.map(tokenize_and_align_labels, batched=True)
XCOLD_test = XCOLD_test.map(tokenize_and_align_labels, batched=True)

from transformers import DataCollatorForTokenClassification
import torch.nn as nn
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
from torchcrf import CRF

class CustomModelWithCRF(AutoModelForTokenClassification):
    def __init__(self, config,num_labels):
        super().from_config(config)
        self.crf = CRF(num_labels)  # num_labels is the number of classes/labels in your task

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.byte())
            return loss
        else:
            return logits

# Initialize the custom model with CRF
model = CustomModelWithCRF.from_pretrained(
   "Rostlab/prot_bert_bfd", 
    num_labels=3, 
    id2label=id2label, 
    label2id=label2id,
)

def find_bi_segments(sequence):
    bi_segments = []  # List to hold the start and end indexes of BI segments
    start_index = None  # Start index of a BI segment

    # Iterate over the sequence with index
    for i, char in enumerate(sequence):
        # Check for 'B' to mark the start of a BI segment
        if char == 'I' and start_index is None:
            start_index = i
        # If the current character is not 'I' and we have a start index, end the segment
        elif char != 'I' and start_index is not None:
            bi_segments.append((start_index, i - 1))  # Add the segment (start_index, end_index)
            start_index = None  # Reset start_index for the next BI segment

    # If the last character is 'I', close the last BI segment
    if start_index is not None:
        bi_segments.append((start_index, len(sequence) - 1))

    return bi_segments

import numpy as np

def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != -100:  # ignore_index is -100
                out_label_list[i].append(id2label[label_ids[i][j]].replace("PAD", "O"))
                preds_list[i].append(id2label[preds[i][j]].replace("PAD", "O"))
                
    return preds_list, out_label_list

def compute_metrics(p):
    def reconstruct_list(flattened_list):
        reconstructed_list = []
        index = 0
        sizes=[644, 455, 556, 474, 499, 465, 449, 473, 465, 449, 496, 351, 178, 121, 476, 605, 529, 471, 478, 487, 770, 273, 204, 150, 128, 133, 121, 604, 617, 164, 427, 323, 379, 450, 452, 248, 113, 113, 432, 258, 277, 248, 749, 588, 431, 443, 202, 463, 498, 260, 284, 262, 554, 269, 558, 522, 663, 176, 438, 417, 302, 746, 185, 401, 572, 79, 806, 283, 283, 264, 291, 107, 244, 400, 894, 306, 431, 127, 230, 575, 258, 796, 202, 703, 1330, 1210, 296, 235, 228, 279, 637, 295, 414, 760, 259, 97, 273, 296, 287, 141, 321, 328, 409, 290, 301, 53, 296, 308, 1827, 319, 142, 99, 422, 450, 326, 412, 228, 413, 109, 381, 907, 206, 156, 188, 448, 272, 514, 477, 323, 273, 121, 216, 1367, 323, 469, 371, 381, 383, 378, 348, 348, 364, 364, 637, 244, 281, 219, 237, 236, 267, 227, 237, 408, 573, 551, 218, 41, 844, 344, 615]
        for size in sizes:
            # Slice the flattened list from the current index to the index + size
            sublist = flattened_list[index:index + size]
            reconstructed_list.append(sublist)
            index += size  # Update the index to the next starting point
        return reconstructed_list
        
    
    predictions, labels=p
    predictions, labels=align_predictions(predictions, labels)
    predictions=reconstruct_list([item for sublist in predictions for item in sublist])
    labels=reconstruct_list([item for sublist in labels for item in sublist])
    def calculate_overlap(interval1, interval2):
        """Calculate the overlap between two intervals."""
        start1, end1 = interval1
        start2, end2 = interval2
        if start1 <= end2 and start2 <= end1:
            return max(0, min(end1, end2) - max(start1, start2))
        return 0

    def perfect_matching_percentage(nested_list1, nested_list2):
        # Track the number of perfectly matched pairs
        perfect_matches = 0

        # Iterate over pairs of sequences from both lists
        for seq1, seq2 in zip(nested_list1, nested_list2):
            intervals1 = find_bi_segments(seq1)
            intervals2 = find_bi_segments(seq2)
            matched = True  # Assume a perfect match initially

            # Check if the lengths of interval lists are the same
            if len(intervals1) != len(intervals2):
                matched = False
            else:
                # Check every pair of intervals for the overlap criterion
                for interval1, interval2 in zip(intervals1, intervals2):
                    if calculate_overlap(interval1, interval2) < 3:
                        matched = False
                        break  # No need to check further for this pair of sequences
            
            # If all intervals matched perfectly, increment the counter
            if matched:
                perfect_matches += 1

        # Calculate and return the percentage of perfect matches
        total_pairs = len(nested_list1)  # Assuming both lists have the same length
        return (perfect_matches / total_pairs)
    

    predictions= [item for sublist in predictions for item in sublist]
    labels= [item for sublist in labels for item in sublist]
    labels=find_bi_segments(labels)
    predictions=find_bi_segments(predictions)


    def match_intervals(predicted_intervals, ground_truth_intervals):
        """Match predicted intervals with ground truth intervals based on the overlap criteria."""
        matches = []
        used_gt = set()
        for pred_interval in predicted_intervals:
            for gt_index, gt_interval in enumerate(ground_truth_intervals):
                if gt_index not in used_gt and calculate_overlap(pred_interval, gt_interval) >= 3:
                    matches.append((pred_interval, gt_interval))
                    used_gt.add(gt_index)
                    break
        return matches

    def evaluate_intervals(predicted_intervals, ground_truth_intervals):
        """Evaluate the precision, recall, and F1 score of the predicted intervals against the ground truth."""
        matched_intervals = match_intervals(predicted_intervals, ground_truth_intervals)
        precision = len(matched_intervals) / len(predicted_intervals) if predicted_intervals else 0
        recall = len(matched_intervals) / len(ground_truth_intervals) if ground_truth_intervals else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1
    # Return the aggregated metrics、
    precision, recall, f1 = evaluate_intervals(predictions, labels)
    pr,l=p
    pr,l= align_predictions(pr,l)
    qok=perfect_matching_percentage(pr, l)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        'qok':qok
    }

training_args = TrainingArguments(    
    output_dir="./114514",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=XCOLD,
    eval_dataset=XCOLD_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics

)
#10fold cross validation
from sklearn.model_selection import KFold

# Assuming XCOLD and XCOLD_test are your original training and testing datasets
# You need to combine them into one dataset for cross-validation
full_dataset = XCOLD

# Prepare KFold cross-validation with 10 splits
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Convert the Hugging Face dataset to a format compatible with sklearn's KFold
# This step is necessary because KFold expects a list or a numpy array
examples = [example for example in full_dataset]

# Collect the evaluation metrics for each fold
metrics = []

import matplotlib.pyplot as plt
'''from transformers import TrainerCallback

# Define a callback to capture the loss at the end of each epoch
class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.epoch_losses = []

    def on_epoch_end(self, args, state, control, **kwargs):
        # Log the loss at the end of each epoch
        logs = kwargs.get("logs", {})
        if "loss" in logs:
            self.epoch_losses.append(logs["loss"])'''

# Initialize an empty list to store losses for each fold
losses_per_fold = []

for fold, (train_ids, test_ids) in enumerate(kfold.split(examples)):
    print(f"Training fold {fold + 1}/10...")
    
    # Split the dataset into training and validation for the current fold
    train_dataset = Dataset.from_list([examples[i] for i in train_ids])
    test_dataset = Dataset.from_list([examples[i] for i in test_ids])
    
    # Preprocess the data
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

    # Initialize the model for each fold to start fresh
    model = AutoModelForTokenClassification.from_pretrained(
        "Rostlab/prot_bert_bfd", num_labels=3, id2label=id2label, label2id=label2id
    )

    # Prepare the callback
    loss_logger = LossLoggingCallback()

    # Initialize the Trainer for the current fold
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[loss_logger]  # Add the loss logging callback
    )

    # Train and evaluate the model on the current fold
    trainer.train()
    trainer.evaluate()

    # Store the losses for this fold
    losses_per_fold.append(loss_logger.epoch_losses)
# Plotting the loss for each fold
plt.figure(figsize=(12, 8))
for idx, losses in enumerate(losses_per_fold):
    epochs = list(range(1, len(losses) + 1))
    plt.plot(epochs, losses, marker='o', label=f'Fold {idx + 1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch in Each Fold')
plt.legend()
plt.show()
plt.savefig()
