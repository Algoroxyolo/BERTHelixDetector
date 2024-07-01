def find_blank_line_numbers(input_filename):
    # Open the file and read line by line
    with open(input_filename, 'r', encoding='utf-8') as file:
        line_number = 0
        blank_line_numbers = []
        
        for line in file:
            line_number += 1
            # Check if the line is blank (only contains newline)
            if line == '\n':
                blank_line_numbers.append(line_number)
    
    return blank_line_numbers

# Usage
input_file = 'test-noi.txt'
blank_lines = find_blank_line_numbers(input_file)
print("Blank lines are found at line numbers:", blank_lines)


def count_lines_per_entry(input_filename):
    # Read the entire file
    with open(input_filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content into entries
    entries = content.strip().split('\n\n')  # split by double newlines
    
    # Calculate the number of lines in each entry
    entry_line_counts = [entry.count('\n') + 1 for entry in entries]
    return entry_line_counts

# Usage
input_file = 'test-noi.txt'
entry_line_counts = count_lines_per_entry(input_file)
print("Number of lines in each entry:", entry_line_counts)
def reconstruct_list(flattened_list, sizes):
    reconstructed_list = []
    index = 0
    for size in sizes:
        # Slice the flattened list from the current index to the index + size
        sublist = flattened_list[index:index + size]
        reconstructed_list.append(sublist)
        index += size  # Update the index to the next starting point
    return reconstructed_list

# Example Usage
flattened_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
sizes = [3, 2, 4]  # This means the first sublist should have 3 items, the second 2 items, and the third 4 items

new_list = reconstruct_list(flattened_list, sizes)
print("Reconstructed List of Lists:", new_list)
