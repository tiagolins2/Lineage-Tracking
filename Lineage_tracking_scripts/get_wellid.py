import csv
import os 
# Function to extract information from the address
def extract_info(address):
    # Split the address by "/"
    parts = address#.split("/")
    # Get the values from specific indices
    cam_info = parts[-45:-31]
    plate_info = parts[-14:-8]
    last_chars = parts[-7:]
    return cam_info, plate_info, last_chars

# Read addresses from a text file
directory = input("Enter the full path of the directory containing addresses.txt: ")

# Combine the directory with the file name
addresses_file = os.path.join(directory, 'output_paths.txt')

if os.path.exists(addresses_file):
    # Read addresses from the text file
    with open(addresses_file, 'r') as file:
        addresses = file.readlines()

    # Combine the directory with the output CSV file name
    output_csv_file = os.path.join(directory, 'output.csv')

# Create and write to the CSV file
    with open(output_csv_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header row
        writer.writerow(['Address', 'Cam Info', 'Plate Info', 'Last Chars', 'ID'])
        
        # Dictionary to store unique combinations and IDs
        combination_ids = {}
        current_id = 1
        
        # Iterate through addresses and extract information
        for address in addresses:
            address = address.strip()  # Remove whitespace characters such as '\n'
            cam_info, plate_info, last_chars = extract_info(address)
            combination = (cam_info, plate_info, last_chars)
            
            # Check if combination already exists in the dictionary
            if combination in combination_ids:
                id_ = combination_ids[combination]
            else:
                id_ = current_id
                combination_ids[combination] = id_
                current_id += 1
            
            writer.writerow([address, cam_info, plate_info, last_chars, id_])

    print(f"CSV file generated successfully in {output_csv_file}")
else:
    print("addresses.txt file not found in the specified directory.")
