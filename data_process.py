def extract_tm_sequences_to_bio(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        while True:
            header = infile.readline().strip()
            if not header:  # End of file
                break
            sequence = infile.readline().strip()
            annotation = infile.readline().strip()
            
            if "|TM" in header or True:
                prev_char = ''
                for i, char in enumerate(annotation):
                    if char == 'M':
                        if prev_char != 'M':  # Beginning of a transmembrane region
                            tag = 'B'
                        else:  # Inside a transmembrane region
                            tag = 'I'
                    else:
                        tag = 'O'  # Outside a transmembrane region
                    outfile.write(f"{sequence[i]}\t{tag}\n")
                    prev_char = char
                outfile.write("\n")  # Separate entries by an empty line



# Example usage:
input_file_path = "test-raw.txt"
output_file_path = "test.txt"
extract_tm_sequences_to_bio(input_file_path, output_file_path)
