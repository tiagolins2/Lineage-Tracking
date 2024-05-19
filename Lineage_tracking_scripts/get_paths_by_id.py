import csv
import os
def read_addresses_by_id(csv_file_path):
    # Check if the CSV file exists
    if not csv_file_path.endswith('.csv') or not os.path.exists(csv_file_path):
        print("Invalid CSV file path or file does not exist.")
        return
    
    # Dictionary to store addresses by ID
    addresses_by_id = {}
    
    # Open the CSV file for reading
    with open(csv_file_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        
        # Skip the header row
        next(reader)
        
        # Read each row in the CSV file
        for row in reader:
            address, cam_info, plate_info, last_chars, id_ = row
            
            # Convert ID to integer
            id_ = int(id_)
            
            # Check if ID already exists in the dictionary
            if id_ in addresses_by_id:
                addresses_by_id[id_].append((address, cam_info, plate_info, last_chars))
            else:
                addresses_by_id[id_] = [(address, cam_info, plate_info, last_chars)]
    
    return addresses_by_id

# Example usage
#csv_file_path = input("Enter the full path of the output.csv file: ")
#addresses_by_id = read_addresses_by_id(os.path.join(csv_file_path, 'output.csv'))

# Print addresses by ID
#for id_, addresses_info in addresses_by_id.items():
#    print(f"ID: {id_}")
#    for address_info in addresses_info:
#        address, cam_info, plate_info, last_chars = address_info
#        print(f"Address: {address}, Cam Info: {cam_info}, Plate Info: {plate_info}, Last Chars: {last_chars}")
#    print()
