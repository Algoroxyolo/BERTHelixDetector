import pickle
with open('lorenzotesting.pkl', 'rb') as f:
    predictedLabels,predictedProteins,totalLabels,observedProteins = pickle.load(f)


def compute_metrics(labels, predictions):
    def find_bi_segments(sequence):
        bi_segments = []  # List to hold the start and end indexes of BI segments
        start_index = None  # Start index of a BI segment

        # Iterate over the sequence with index
        for i, integer in enumerate(sequence):
            # Check for 'B' to mark the start of a BI segment
            if integer == 1 and start_index is None:
                start_index = i
            # If the current character is not 'I' and we have a start index, end the segment
            elif integer != 1 and start_index is not None:
                bi_segments.append((start_index, i - 1))  # Add the segment (start_index, end_index)
                start_index = None  # Reset start_index for the next BI segment

        # If the last character is 'I', close the last BI segment
        if start_index is not None:
            bi_segments.append((start_index, len(sequence) - 1))

        return bi_segments
    
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

    qok=perfect_matching_percentage(predictions, labels)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        'qok':qok
    }
print(compute_metrics(predictedProteins,observedProteins))