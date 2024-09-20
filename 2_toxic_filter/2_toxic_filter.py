import os
import json
import chardet
import argparse
from pathlib import Path

def load_jsonl(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return [json.loads(l) for l in f]

class CorpusFilter:
    def __init__(self, directory_path):
        self.sensitive_keywords = self.load_sensitive_keywords(directory_path)
    
    def detect_encoding(self, file_path):
        with open(file_path, 'rb') as file:
            raw_data = file.read(5000)
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        return encoding
    
    def load_sensitive_keywords(self, directory_path):
        # Load sensitive keywords from all .txt files in the specified directory
        sensitive_keywords = set()
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                encoding = self.detect_encoding(file_path)
                with open(file_path, 'r', encoding=encoding) as file:
                    for line in file:
                        keyword = line.strip().rstrip(',')
                        if keyword:
                            sensitive_keywords.add(keyword)
        return list(sensitive_keywords)
    
    def is_sensitive(self, text):
        for keyword in self.sensitive_keywords:
            if keyword in text:
                return True
        return False
    
    def filter_corpus(self, input_file_path, output_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as input_file, \
             open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                try:
                    data = json.loads(line)
                    text = data.get(args.text_column, '')
                    if not self.is_sensitive(text):
                        output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                except json.JSONDecodeError:
                    continue  # Ignore lines with parsing errors



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # The default input and output are jsonl files
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--text_column', type=str)
    args = parser.parse_args()

    directory_path = Path(__file__).parent / "sensitive_words"
    filter = CorpusFilter(directory_path)
    data = load_jsonl(args.input_path)
    filter.filter_corpus(args.input_path, args.output_path)   

