import argparse
import os
from corpus_evaluator import corpus_quality_measure_fn
from corpus_eval_visulization import scores_visualization


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=1234)

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_num", type=int, default=10)
    parser.add_argument("--text_column", type=str)
    parser.add_argument("--tiktoken_cache", type=str)

    parser.add_argument("--eval_path", type=str)

    parser.add_argument("--figure_dir", type=str)

    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--organization", type=str)
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()

    # args.eval_path = args.figure_dir + "/result.jsonl"

    tiktoken_cache_dir = args.tiktoken_cache
    os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

    corpus = corpus_quality_measure_fn(
        data_path=args.data_path,
        eval_path=args.eval_path,
        data_num=args.data_num,
        text_column=args.text_column,
        model=args.model,
        api_key=args.api_key,
        organization=args.organization,
        num_proc=args.num_proc,)

    scores_visualization(corpus, args.text_column, args.figure_dir)

