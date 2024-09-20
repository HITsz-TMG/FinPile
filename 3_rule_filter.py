import json
import argparse
import ftfy
import regex
from langdetect import detect
from tqdm import tqdm
import opencc

def load_jsonl(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return [json.loads(l) for l in f]

class RuleFilter:
    def __init__(self):
        self.OPENCC_CONVERTER = opencc.OpenCC('t2s.json')
        self.punctuation_unicode = {
            '，': ',',
            '。': '.',
            '、': ',',
            '„': '"',
            '”': '"',
            '“': '"',
            '«': '"',
            '»': '"',
            '１': '"',
            '」': '"',
            '「': '"',
            '《': '"',
            '》': '"',
            '´': "'",
            '∶': ':',
            '：': ':',
            '？': '?',
            '！': '!',
            '（': '(',
            '）': ')',
            '；': ';',
            '–': '-',
            '—': ' - ',
            '．': '. ',
            '～': '~',
            '’': "'",
            '…': '...',
            '━': '-',
            '〈': '<',
            '〉': '>',
            '【': '[',
            '】': ']',
            '％': '%',
            '►': '-',
        }
        self.various_whitespaces = {
            ' ', '	', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
            ' ', ' ', ' ', '　', '​', '‌', '‍', '⁠', '￼', ''
        }
        
    def handle(self, text):
        # unicode
        text = ftfy.fix_text(text, normalization="NFC")
        # language filter
        if detect(text) != args.language:
            return None
        
        # Standardization of Punctuation
        text = ''.join([
            self.punctuation_unicode.get(c, c) for c in text
        ])
        # Standardization of Whitespace
        text =  ''.join([
            char if char not in self.various_whitespaces else ' ' for char in text
        ])
        
        # Replace all matched consecutive punctuation with a single punctuation
        pattern = r'(\p{P})\1+'
        text = regex.sub(pattern, r'\1', text)
        text = text.strip()
        
        # Filter out texts with too high a punctuation ratio and too short a text length
        punctuation_count = len(regex.findall(r'\p{P}', text))
        total_chars = len(text)
        punctuation_ratio = punctuation_count / total_chars
        if punctuation_ratio > args.punctuation_ratio_threshold or len(text) < args.text_length_threshold:
            return None

        
        # Convert Traditional Chinese Characters to Simplified Chinese
        return self.OPENCC_CONVERTER.convert(text)
    
    def filter(self, input_file_path, output_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as input_file, \
             open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                try:
                    data = json.loads(line)
                    text = data.get(args.text_column, '')
                    result = self.handle(text)
                    if result:
                        data[args.text_column] = result
                        output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                except json.JSONDecodeError:
                    continue  # Ignore lines with parsing errors



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # The default input and output are jsonl files
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--text_column', type=str)
    parser.add_argument('--language', type=str)
    parser.add_argument('--punctuation_ratio_threshold', type=float, default=0.5)
    parser.add_argument('--text_length_threshold', type=int, default=128)
    args = parser.parse_args()

    filter = RuleFilter()
    filter.filter(args.input_path, args.output_path)
