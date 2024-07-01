from transformers import AutoTokenizer
import torch
'''import signal
def handler_function(a,b):
    print("in handler")
    print(torch.cuda.is_available())
signal.signal(signal.SIGALRM,handler_function)
signal.alarm(5)
print(torch.cuda.is_available())'''

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

model = AutoModelForTokenClassification.from_pretrained(
    "Rostlab/prot_bert_bfd", num_labels=3, id2label=id2label, label2id=label2id,
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
    predictions, labels=p
    predictions, labels=align_predictions(predictions, labels)
    # Flatten the lists
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_labels = [item for sublist in labels for item in sublist]



    def calculate_overlap(interval1, interval2):
        start1, end1 = interval1
        start2, end2 = interval2
        return max(0, min(end1, end2) - max(start1, start2) + 1)

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
    def calculate_overlap(interval1, interval2):
        """Calculate the overlap between two intervals."""
        start1, end1 = interval1
        start2, end2 = interval2
        if start1 <= end2 and start2 <= end1:
            return max(0, min(end1, end2) - max(start1, start2))
        return 0

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


# Note: Make sure that the rest of your setup for training and evaluation remains the same.



#model_with_crf = CustomModelWithCRF(model, 3)

training_args = TrainingArguments(    
    output_dir="./114514",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
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

trainer.train()
trainer.evaluate()
def align_predictions2(predictions, label_ids):
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
# After calling trainer.evaluate(), get predictions
predictions = trainer.predict(XCOLD_test)
preds_list, out_label_list = align_predictions2(predictions.predictions, predictions.label_ids)
# Writing predictions and labels to a file
with open('evaluation_results.txt', 'w', encoding='utf-8') as file:
    for labels, preds in zip(out_label_list, preds_list):
        # Convert ID labels back to string labels for writing

        labels_str = [id for id in labels]
        preds_str = [id for id in preds]

        file.write(' '.join(labels_str) + '\n')
        file.write(' '.join(preds_str) + '\n\n')

# Note: This code assumes that 'align_predictions' returns the labels and predictions
# in a format where each item in `preds_list` and `out_label_list` corresponds to the predictions
# and true labels of a single example in the evaluation set.

'''save_directory = "saved_model"  # Provide the directory path where you want to save the model
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)'''

'''print("Model and tokenizer saved to:", save_directory)'''

"""#10fold cross validation
from sklearn.model_selection import KFold
from datasets import concatenate_datasets

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
        "Rostlab/prot_bert_bfd", num_labels=3, id2label=id2label, label2id=label2id,
    )

    # Initialize the Trainer for the current fold
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train and evaluate the model on the current fold
    trainer.train()
    evaluation_results = trainer.evaluate()

    # Store the evaluation metrics from this fold
    metrics.append(evaluation_results)

# Calculate and print the average metrics across all folds
average_metrics = {key: sum(metric[key] for metric in metrics) / len(metrics) for key in metrics[0]}
print("Average metrics across all folds:", average_metrics)
"""