# BWT string
bwt_string = "enwpeoseu$Ilt"

# Step 1: Initialize the matrix with the BWT string characters as the last column
matrix = [[char] for char in bwt_string]

# Step 2: Sort the matrix (initially just the BWT string) to form the first iteration
matrix.sort()

# Repeat inserting and sorting to form the full Burrows-Wheeler matrix
for _ in range(len(bwt_string) - 1):
    # Insert BWT string as a new column to the left of existing matrix
    for i, char in enumerate(bwt_string):
        matrix[i].insert(0, char)
    # Sort the matrix
    matrix.sort()

# The original string is in the row that ends with the '$' character, read from top to bottom.

for line in matrix:
    print(line)