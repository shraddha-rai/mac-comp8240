import pandas as pd
import re


def preprocess_imdb(input_csv, output_txt):
    # Load CSV with text and label columns
    df = pd.read_csv(input_csv)

    # Verify expected columns
    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # Map 0 → negative, 1 → positive
    label_map = {0: "negative", 1: "positive"}
    df['label'] = df['label'].map(label_map)

    # Clean text: remove newlines/tabs and collapse whitespace
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].apply(lambda t: re.sub(r'\s+', ' ', t.replace('\t', ' ')).strip())

    with open(output_txt, 'w', encoding='utf-8', newline='') as f:
        for _, row in df.iterrows():
            f.write(f'{row["label"]}\t{row["text"]}\n')
    print(f"Text file saved to {output_txt}")


# Example usage
if __name__ == "__main__":
    preprocess_imdb('data_downloaded/movie.csv', 'new_data/subj/test.txt')
