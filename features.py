import os
import csv

# Path to the input text file containing word information
input_file_path = '/Users/krishilparikh/Desktop/Proj/words_new.txt'
# Path for the output labels.csv file
output_file_path = '/Users/krishilparikh/Desktop/Proj/database/labell.csv'

# Initialize the list to store label rows
labels = []

# Write header information to the labels.csv file
header = ['word_id', 'result', 'graylevel', 'x', 'y', 'width', 'height', 'grammatical_tag', 'transcription']
labels.append(header)

# Read the word information from the input text file
with open(input_file_path, 'r') as f:
    for line in f:
        # Strip any leading/trailing whitespace characters
        line = line.strip()
        
        # Split the line into components
        components = line.split()

        # Initialize variables for each component
        word_id = result = graylevel = num_components = x = y = width = height = grammatical_tag = transcription = ''

        # Extract components based on available information
        if len(components) >= 10:
            word_id = components[0]      
            result = components[1]       
            graylevel = components[2]    
            num_components = components[3] 
            x = components[4]            
            y = components[5]            
            width = components[6]        
            height = components[7]       
            grammatical_tag = components[8] 
            transcription = components[9] 
        elif len(components) >= 9:
            word_id = components[0]      
            result = components[1]       
            graylevel = components[2]    
            num_components = components[3] 
            x = components[4]            
            y = components[5]            
            width = components[6]        
            height = components[7]       
            grammatical_tag = components[8] 
            transcription = ''  # Missing transcription
        elif len(components) >= 8:
            word_id = components[0]      
            result = components[1]       
            graylevel = components[2]    
            num_components = components[3] 
            x = components[4]            
            y = components[5]            
            width = components[6]        
            height = components[7]       
            grammatical_tag = ''
            transcription = ''
        else:
            print(f"Warning: Line skipped due to insufficient components: {line}")

        # Format the row and add it to the list, regardless of missing components
        row = [word_id, result, graylevel, num_components, x, y, width, height, grammatical_tag, transcription]
        labels.append(row)

# Sort the labels list based on the first column (word_id), excluding header
sorted_labels = sorted(labels[1:], key=lambda x: x[0])  

# Write the sorted labels to the labels.csv file
with open(output_file_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows([header] + sorted_labels)  # Write header and all data

print(f"Labels file created at: {output_file_path}")
