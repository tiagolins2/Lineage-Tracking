import os
import pandas as pd

# Get the location of the folder containing the Excel file
folder_location = input("Enter the full path of the folder containing the Excel file: ")

# Combine the folder location with the Excel file name
excel_file = os.path.join(folder_location, 'full_data.csv')  # Replace with the actual Excel file name

# Check if the Excel file exists
if os.path.exists(excel_file):
    # Load the Excel file
    df = pd.read_csv(excel_file)
    print(df)
    output_file = os.path.join(folder_location, 'output_paths.txt')

    # Define a function to check if a file exists and append its path to the output file
    def append_path_txt(folder_location, folder, file_name):
        file_path = os.path.join(folder_location + "/" + folder, file_name + '.png')
        if os.path.exists(file_path):
            print(folder)
            with open(output_file, 'a') as txt_file:
                txt_file.write(file_path + '\n')

    # Iterate through each row in the dataframe and check file existence
    for index, row in df.iterrows():
        folder = str(row['filename'])
        file_name = str(row['well_id'])
        
        append_path_txt(folder_location, folder, file_name)
else:
    print("Excel file not found in the specified location.")
