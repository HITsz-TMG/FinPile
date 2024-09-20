import re
import random
from collections import OrderedDict
import json
import tiktoken
import openai
from openai import OpenAI
from datasets import Dataset, load_dataset

# add http proxy
# import os
# os.environ["http_proxy"] = "http://127.0.0.1:10809"
# os.environ["https_proxy"] = "http://127.0.0.1:10809"

PROMPT = """你是一个语料评价专家，负责对单条语料（通常是一段自然语言文本）的质量进行打分以用于大语言模型的预训练
你的评价标准是：
语言质量(0-10分):考察语料的语法、拼写、词汇是否正确,语言表达是否流畅。语言质量高的语料利于模型学习语言规则,可以得高分。得分依据:语法和拼写正确(2分),词汇丰富(2分),表达流畅(2分),长难句或生僻词出现(2分),语言总体复杂(2分)。

信息量(0-10分):考察语料所包含的知识量和概念量。信息量大的语料有利于模型学习丰富知识,可以得高分。得分依据:包含专业知识或生僻概念(3分),篇幅较长或讨论多个话题(3分),详尽叙述某一话题(2分),提供新的信息或见解(2分)。 

新颖性(0-10分):考察语料中的新奇词汇、新信息或新思想对模型理解范围的扩展作用。新颖性高的语料可以得高分。得分依据:包含新词或新概念(3分),提供新信息或新见解(3分),采用新角度或新形式表达观点(2分),创造新的词或短语(2分)。

连贯性(0-10分): 主题明确,观点连贯,论证严谨,构成完整论述(3分);主题基本清晰,且论证严谨。(3分) 各部分同属同一话题，构成连贯整体(4分)。

纯净度(0-10分):考察语料含有无关信息如广告、营销、垃圾信息的数量，含此类信息少而大部分内容都与主题相关的语料可以得高分。得分依据:主要内容表达完整(3分),垃圾信息含量少(3分)，完全没有垃圾信息(4分)

通过以上评价标准，你将对下面的语料进行打分：
【语料开始】

{corpus}

【语料结束】

请先分条给出评价理由，再给出对应分数并格式化输出。
示例：
【语言质量】:语法和拼写基本正确，词汇较丰富，表达流畅，出现生僻词如“幽灵枪”和长句，语言较复杂。【分数】8
【信息量】:涉及专业领域知识如各类枪支、美国控枪法案等，讨论多个话题如美国枪支文化与政策、美国枪支暴力现状等，详尽论述美国枪支状况，提供大量数据与信息。【分数】9
【新颖性】:出现新词“幽灵枪”和新概念如“极端枪支文化”，从政治经济角度揭示美国枪支问题新原因，以全新的角度解析美国枪支文化。【分数】8
【连贯性】:文中各部分紧密衔接，从美国枪支政策演变到枪支问题分析，再到政治经济因素剖析，行文逻辑清晰，段落结构明确。【分数】9
【纯净度】:文中的主要内容表达完整，大部分文本都与主题相关，但是结尾含有推广引流信息，不过垃圾信息含量较少。【分数】7

输出："""

all_aspects = ["语言质量", "信息量", "新颖性", "连贯性", "纯净度"]

tokenizer = tiktoken.get_encoding('cl100k_base')


def read_data(data_path: str, data_num: int) -> Dataset:
    dataset = load_dataset("json", data_files=[data_path], split="train", keep_in_memory=True)

    if data_num is not None:
        data_num = min(data_num, len(dataset))
    random_indices = random.sample(range(len(dataset)), data_num)

    return dataset.select(random_indices)


def cut_corpus(text, max_len=1000):
    text_tokens = tokenizer.encode(str(text).strip())
    if len(text_tokens) > max_len:
        text_readable = False
        text_tokens = text_tokens[:max_len]
        while not text_readable and len(text_tokens) > 1:
            try:
                text = tokenizer.decode(text_tokens)
                text_readable = True
            except:
                text_tokens = text_tokens[:-1]
    return text


def call_openai_func(instruction: str, model: str = "gpt-3.5-turbo-1106", api_key: str = None, organization: str = None) -> str:
    
    openai.api_key = api_key
    openai.organization = organization

    client = OpenAI(api_key=api_key, organization=organization)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return completion.choices[0].message.content



def extract_result(text: str) -> dict:
    pattern = r'【(.*?)】:(.*?)【分数】(\d+)\n'

    matches = re.findall(pattern, text+"\n")

    result = OrderedDict({aspect: {"reason": "", "score": -1} for aspect in all_aspects})
    assert len(matches) == len(all_aspects)
    for match in matches:
        aspect = match[0]
        reason = match[1]
        score = match[2]
        assert aspect in all_aspects
        result[aspect] = {"reason": reason, "score": int(score)}

    return result

def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for item in data:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            
def corpus_quality_measure_fn(
        data_path: str,
        eval_path: str = None,
        data_num: int = None,
        text_column: str = "text",
        model: str = "gpt-3.5-turbo-1106",
        api_key: str = None,
        organization: str = None,
        num_proc: int = 1,):

    def eval_single_item(obj):
        text = obj[text_column]
        instruction = PROMPT.format(corpus=cut_corpus(text))

        try:
            response = call_openai_func(instruction, model, api_key, organization)
            result = extract_result(response)
        except Exception as e:
            print("Error")
            print(e)
            result = OrderedDict({aspect: {"reason": "", "score": -1} for aspect in all_aspects})

        obj["quality"] = result
        return obj

    corpus = read_data(data_path, data_num)
    corpus = corpus.map(eval_single_item, num_proc=num_proc)

    if eval_path is not None:
        save_jsonl(corpus, eval_path)
        # corpus.to_json(eval_path, batch_size=128, force_ascii=False)

    return corpus
