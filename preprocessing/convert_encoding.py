import pandas as pd
import os

def convert_encoding(input_path, output_path, source_encoding="latin1", target_encoding="utf-8"):
    try:
        # Read using source encoding (e.g., latin1)
        df = pd.read_csv(input_path, encoding=source_encoding)

        # Save using target encoding (e.g., utf-8)
        df.to_csv(output_path, index=False, encoding=target_encoding)

        print(f"Successfully converted '{input_path}' to UTF-8 and saved as '{output_path}'.")
    except Exception as e:
        print(f"Failed to convert file '{input_path}': {e}")

# Example usage
if __name__ == "__main__":
    input_file = "labeled_data/McDonald_s_Reviews.csv"
    output_file = "labeled_data/McDonald_s_Reviews_utf8.csv"
    convert_encoding(input_file, output_file)
