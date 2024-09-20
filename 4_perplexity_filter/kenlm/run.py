from model import KenlmModel
import json
import jsonlines
import argparse
from tqdm import tqdm

def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for item in data:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")

def read_jsonl(input_path):
    output_data = []
    with open(input_path, 'r+', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            output_data.append(item)
    return output_data

def perplexity_filter(input_path, output_path):
    input_data = read_jsonl(input_path)
    filtered_data = []

    for tmp in tqdm(input_data):
        score = model.get_perplexity(tmp[args.text_column])
        if score <= 2095:
            filtered_data.append(tmp)

    save_jsonl(filtered_data, output_path)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to input file(jsonl).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output file(jsonl).",
    )
    parser.add_argument('--text_column', type=str)
    parser.add_argument('--language', type=str, help="zh or en")
    args = parser.parse_args()
    
    # model taken from https://huggingface.co/edugp/kenlm
    model = KenlmModel.from_pretrained("kenlm/wikipedia", args.language)
    perplexity_filter(args.input_path, args.output_path)
    print('Perplexity Filter Done!')

