# FinPile
Data and tools for generating and inspecting FinPile, a safe, high-quality, open-sourced Chinese financial corpus.

## 🌟 Environment
Our recommended Python version is **3.11.4**. 
```
pip install -r requirements.txt
```

## 🧩 Data Preprocessing

### 1. Remove personal information
This step completes the removal of personal information such as IP addresses, emails, and phone numbers.
#### Example usage
```
python 1_pii.py \
    --input_path input.jsonl \
    --output_path output.jsonl \
    --text_column text \
    --num_proc 4 \
    --batch_size 100
```

### 2. Sensitive Words
To avoid the inclusion of toxic content in the training data, one approach is to filter out texts that contain specific sensitive keywords. You need to store the ***txt*** files containing sensitive words in `2_toxic_filter/sensitive_words`.
#### Example usage
```
python 2_toxic_filter/2_toxic_filter.py \
    --input_path input.jsonl \
    --output_path output.jsonl \
    --text_column text \
```

### 3. Rule Filtering
This step completes multiple rule-based data filtering.
- Language Filtering: Retain only text data in a specific language (***zh-cn*** or ***en***).
- Punctuation and whitespace consistency processing: Unify Chinese and English punctuation within the text, and standardize different types of whitespace characters as well.
- Deduplication of consecutive punctuation: Replace all matched consecutive punctuation marks with a single punctuation mark.
- Punctuation Ratio Filtering: Filter out texts with too high a punctuation ratio.
- Data Length Filtering: Filter out text data that is too short.
#### Example usage
```
python 3_rule_filter.py \
    --input_path input.jsonl \
    --output_path output.jsonl \
    --text_column text \
    --language zh-cn \
    --punctuation_ratio_threshold 0.5 \
    --text_length_threshold 128 \
```

### 4. Perplexity Filtering
You need to first download the model from the [(address)](https://huggingface.co/edugp/kenlm), and then modify the corresponding model path in the following line in `4_perplexity_filter/kenlm/run.py`.
```python
model = KenlmModel.from_pretrained("kenlm/wikipedia", args.language) #language = zh or en
```
#### Example usage
```
python 4_perplexity_filter/kenlm/run.py \
    --input_path input.jsonl \
    --output_path output.jsonl \
    --text_column text \
    --language zh \
```

### 5. Exact Deduplication
Deduplicate identical text entries in the dataset.
#### Example usage
```
python 5_text_dedup/5_clean.py \
    --input_path input.jsonl \
    --output_path output.jsonl \
    --text_column text \
    --cache cache_dir \
    --num_proc 2 \
    --batch_size 100
```

### 6. Fuzzy Deduplication
Deduplicate similar texts in the dataset.
#### Example usage
```
python 6_text_dedup/text_dedup/minhash.py \
    --input_path input.jsonl \
    --output_path output.jsonl \
    --column text \
    --cache_dir cache_dir \
    --threshold 0.8 \
    --false_positive_weight 0.5 \
    --false_negative_weight 0.5 \
```


## ⚡️ Data Evaluation
We evaluate each piece of data from the following aspects:
- Language Quality (0-10 points): This examines whether the data is grammatically correct, spelled correctly, uses appropriate vocabulary, and if the expression is fluent. High language quality aids the model in learning language rules, resulting in a higher score. **Scoring criteria**: correct grammar and spelling (2 points), rich vocabulary (2 points), fluent expression (2 points), use of complex sentences or rare words (2 points), and overall language complexity (2 points).

- Information Content (0-10 points): This measures the amount of knowledge and concepts contained in the data. Data with high information content helps the model learn rich knowledge, leading to a higher score. **Scoring criteria**: includes specialized knowledge or obscure concepts (3 points), longer length or discussion of multiple topics (3 points), detailed discussion of a single topic (2 points), and providing new information or insights (2 points).

- Novelty (0-10 points): This evaluates the extent to which new vocabulary, information, or ideas in the data expand the model's understanding. Data with high novelty can receive higher scores. **Scoring criteria**: includes new words or concepts (3 points), provides new information or insights (3 points), presents ideas from new perspectives or in new forms (2 points), and creates new words or phrases (2 points).

- Coherence (0-10 points): **Scoring criteria**: This assesses whether the data has a clear theme, coherent arguments, and rigorous reasoning, forming a complete discussion (3 points); a mostly clear theme with rigorous reasoning (3 points); all parts belong to the same topic, forming a coherent whole (4 points).

- Purity (0-10 points): This evaluates the amount of irrelevant information, such as ads, marketing, or spam, in the data. Data with little to no such information and content that mostly relates to the topic can score higher. **Scoring criteria**: the main content is fully expressed (3 points), low spam content (3 points), and no spam content at all (4 points).

#### Example usage
```
python 7_DataAnalysis/eval_pipeline.py \
    --data_path input.jsonl \
    --eval_path output.jsonl \
    --text_column text \
    --tiktoken_cache cache_dir \
    --figure_dir figure_dir \
    --model gpt-3.5-turbo-1106 \
    --api_key xxxx \
    --organization xxxx \
    --num_proc 1 \
```

