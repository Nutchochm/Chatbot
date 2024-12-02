import argparse
import pandas as pd
import json
import time
import re

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process an Excel file to generate intents JSON.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the input Excel file.")
    parser.add_argument('--output_file', type=str, required=False, help="Path to save the output JSON file.")
    args = parser.parse_args()

    # Load the Excel file
    df = pd.read_excel(args.file_path)

    intents = []

    # Create a dictionary to store data by tag
    tag_dict = {}

    # Iterate over the rows of the DataFrame
    for _, row in df.iterrows():
        # Ensure 'patterns' and 'responses' are treated as strings
        patterns = str(row["patterns"]) if isinstance(row["patterns"], str) else ""
        responses = str(row["responses"]) if isinstance(row["responses"], str) else ""
        
        # Split patterns and responses using multiple delimiters (comma, period, exclamation mark, question mark)
        patterns_list = [pattern.strip() for pattern in re.split(r'[;,.!?]', patterns) if pattern.strip()]
        responses_list = [response.strip() for response in re.split(r'[;,.!?]', responses) if response.strip()]

        # Check if tag exists in the dictionary
        if row["tag"] not in tag_dict:
            tag_dict[row["tag"]] = {
                "tag": row["tag"],
                "language": {}
            }
        
        # Initialize language keys if not exist
        if row["language"].lower() not in tag_dict[row["tag"]]["language"]:
            tag_dict[row["tag"]]["language"][row["language"].lower()] = {
                "patterns": patterns_list if patterns_list else [patterns],
                "responses": responses_list if responses_list else [responses]
            }
        else:
            # If the language already exists, we append the new patterns and responses
            tag_dict[row["tag"]]["language"][row["language"].lower()]["patterns"].extend(patterns_list)
            tag_dict[row["tag"]]["language"][row["language"].lower()]["responses"].extend(responses_list)

    # Convert the tag dictionary to the final JSON format
    output_json = {
        "intents": list(tag_dict.values())
    }

    # Convert to JSON string with pretty printing
    json_output = json.dumps(output_json, ensure_ascii=False, indent=4)

    # Save to a JSON file
    output_file = args.output_file if args.output_file else f'pretrained/intent_{time.strftime("%Y%m%d")}.json'    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_output)

    # Print the result
    print(f"Output saved to {output_file}")
    print(json_output)

if __name__ == "__main__":
    main()

#python generate_intent.py --file_path pretrained/data/inputs_i_2.xlsx 