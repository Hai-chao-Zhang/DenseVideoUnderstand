import ast
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import openai
import requests
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from openai import OpenAI

import lmms_eval.tasks._task_utils.file_utils as file_utils

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

NUM_SECONDS_TO_SLEEP = 5

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

# Unzip all the zip files to HF HOME cache dir
HF_HOME = os.environ["HF_HOME"]
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "all_test")





# --- 1) 视频路径 loader ---
def lpm_doc_to_visual(doc):
    # `doc["video_path"]` 在 Parquet 里已经存了绝对或相对路径
    return [doc["video_path"]]

# --- 2) 问题拼装 ---
def lpm_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    If the doc was generated from slide-level subtitle QA,
    doc['question'] 已含完整问题。
    """
    pre = lmms_eval_specific_kwargs.get("pre_prompt","")
    post = lmms_eval_specific_kwargs.get("post_prompt","")
    return f"{pre}{doc['question']}{post}"

# --- 3) Ground-truth answer extractor ---
def lpm_doc_to_answer(doc):
    # doc["answer"] 存了我们之前脚本里生成的 subtitles 或 OCR 文本
    return doc["answer"]

# --- 4) GPT-based 评估流程（几乎同 ActivityNetQA） ---
def lpm_process_results(doc, result):
    """
    Args:
      doc: one record from LPM_slides.parquet
      result: [predicted_string]
    Returns:
      dict with gpt_eval_score & gpt_eval_accuracy formats
    """
    question = doc["question"]
    answer   = doc["answer"]
    pred     = result[0]
    # code复用 get_eval & parse_score
    review, _ = get_eval(question, answer, pred, max_tokens=64)
    pred_label, score = parse_score(review)
    common = {
      "video_name": doc["video"],    # slide tag 或 video id
      "question":  question,
      "answer":    answer,
      "pred":      pred,
      "question_id": doc["qid"],
      "type":       doc["type"]
    }
    return {
      "gpt_eval_score":    {**common, "Correctness": pred_label, "score": score},
      "gpt_eval_accuracy": {**common, "Correctness": pred_label, "score": score}
    }

# --- 5) 聚合函数（同样复用 ActivityNetQA 的） ---
def lpm_aggregate_score(results, args):
    yes = sum(1 for r in results if r["Correctness"]=="yes")
    no  = sum(1 for r in results if r["Correctness"]=="no")
    total = sum(r["score"] for r in results)
    return total/len(results) if results else 0

def lpm_aggregate_accuracy(results, args):
    yes = sum(1 for r in results if r["Correctness"]=="yes")
    no  = sum(1 for r in results if r["Correctness"]=="no")
    return yes/(yes+no) if (yes+no)>0 else 0


TRIM_CHAR_LIMIT = 30_000


def get_eval(question, answer, pred, max_tokens: int, retries: int = 5):
    answer = (answer[:TRIM_CHAR_LIMIT] + '...') if len(answer) > TRIM_CHAR_LIMIT else answer
    pred   = (pred[:TRIM_CHAR_LIMIT]   + '...') if len(pred)   > TRIM_CHAR_LIMIT else pred

    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n"
                "## INSTRUCTIONS:\n"
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, "
                "with 5 indicating the highest meaningful match. "
                "Please generate the response only as a Python dictionary string with keys 'pred' and 'score', "
                "where the value of 'pred' is 'yes' or 'no' and the value of 'score' is an INTEGER, not a STRING. "
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION."
            ),
        },
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        backoff = NUM_SECONDS_TO_SLEEP * (2 ** attempt)
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            # 如果是 429 或者其他 5xx/4xx，response.raise_for_status() 会抛出 HTTPError
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            if content:
                return content, data.get("model", "")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            # 专门处理 429
            if status == 429:
                retry_after = e.response.headers.get("Retry-After")
                sleep_time = int(retry_after) if retry_after and retry_after.isdigit() else backoff
                eval_logger.warning(f"Rate limit hit (429). Sleeping for {sleep_time}s (attempt {attempt+1}/{retries})")
                time.sleep(sleep_time)
                continue
            else:
                eval_logger.error(f"HTTP error {status} on attempt {attempt+1}/{retries}: {e}. Backing off {backoff}s.")
                time.sleep(backoff)
                continue
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt+1}/{retries}: {e}. Backing off {backoff}s.")
            time.sleep(backoff)
            continue
        except ValueError as e:
            # JSON decode error
            eval_logger.error(f"JSON decode error on attempt {attempt+1}/{retries}: {e}. Response text: {response.text}")
            time.sleep(backoff)
            continue
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt+1}/{retries}: {e}. Backing off {backoff}s.")
            time.sleep(backoff)
            continue

    eval_logger.error(f"All {retries} attempts failed.")
    return "", ""



# def get_eval(question, answer, pred, max_tokens: int, retries: int = 5):
#     global headers

#     answer = (answer[:TRIM_CHAR_LIMIT] + '...') if len(answer) > TRIM_CHAR_LIMIT else answer
#     pred   = (pred[:TRIM_CHAR_LIMIT]   + '...') if len(pred)   > TRIM_CHAR_LIMIT else pred

#     messages = [
#         {
#             "role": "system",
#             "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
#             "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
#             "------"
#             "##INSTRUCTIONS: "
#             "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
#             "- Consider synonyms or paraphrases as valid matches.\n"
#             "- Evaluate the correctness of the prediction compared to the answer.",
#         },
#         {
#             "role": "user",
#             "content": f"Please evaluate the following video-based question-answer pair:\n\n"
#             f"Question: {question}\n"
#             f"Correct Answer: {answer}\n"
#             f"Predicted Answer: {pred}\n\n"
#             "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
#             "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
#             "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
#             "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}.",
#         },
#     ]

#     payload = {
#         "model": GPT_EVAL_MODEL_NAME,
#         "messages": messages,
#         "temperature": 0,
#         "max_tokens": max_tokens,
#     }

#     for attempt in range(retries):
#         try:
#             response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
#             response.raise_for_status()  # Raises HTTPError for bad responses
#             try:
#                 response_data = response.json()  # Attempt to parse JSON
#             except requests.exceptions.JSONDecodeError:
#                 eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
#                 continue  # Skip to next retry
#             content = response_data["choices"][0]["message"]["content"].strip()
#             if content != "":
#                 return content, response_data["model"]
#         # Handle HTTP errors separately
#         except requests.exceptions.HTTPError as e:
#             eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
#         # Handle other requests-related errors
#         except requests.exceptions.RequestException as e:
#             eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
#         except Exception as e:
#             eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

#         # Handle other unexpected errors
#         if attempt < retries - 1:
#             time.sleep(NUM_SECONDS_TO_SLEEP)
#         else:  # If this was the last attempt, log and return empty
#             eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
#             return "", ""

#     return "", ""


def parse_score(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review = "{" + review.split("{")[1].split("}")[0] + "}"
        review_dict = ast.literal_eval(review)
        # import pdb;pdb.set_trace()
        score_match = review_dict["score"]
        score = int(score_match)
        pred = review_dict["pred"]
        if "yes" in pred.lower():
            pred = "yes"
        elif "no" in pred.lower():
            pred = "no"
        # pred = review_dict.get("pred", "no")
        # score = review_dict.get("score", 0)
        return [pred, score]
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")


def activitynetqa_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary
    """
    try:
        question = doc["question"]
        answer = doc["answer"]
        pred = result[0]

        # Assume get_eval returns a review and the model name, and parse_score parses this review
        review, model_name = get_eval(question, answer, pred, 64)
        scores = parse_score(review)
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        scores = ["no", 0]

    return {
        "gpt_eval_score": {"video_name": doc["video_name"], "question": doc["question"], "answer": doc["answer"], "pred": pred, "question_id": doc["question_id"], "type": doc["type"], "Correctness": scores[0], "score": scores[1]},
        "gpt_eval_accuracy": {"video_name": doc["video_name"], "question": doc["question"], "answer": doc["answer"], "pred": pred, "question_id": doc["question_id"], "type": doc["type"], "Correctness": scores[0], "score": scores[1]},
    }


def activitynetqa_gpt_eval(results, args):
    """
    Process the result file containing predictions, score them using GPT,
    and save the results with added scores and correctness fields to a new file.

    Args:
        result_file_path: path to the JSON file with results to be evaluated
        eval_file_path: path to save the JSON file with evaluated results
    """

    evaluated_results = []

    # Process each result to generate scores
    for data_dict in results:
        try:
            question = data_dict.get("Q", "")
            answer = data_dict.get("A", "")
            pred = data_dict.get("pred", "")

            # Assume get_eval returns a review and the model name, and parse_score parses this review
            review, model_name = get_eval(question, answer, pred, 64)
            scores = parse_score(review)
        except Exception as e:
            eval_logger.error(f"Error for Question ID: {data_dict.get('question_id', 'Unknown')}: {e}")
            review = "Failed to Get a Proper Review."
            model_name = "Failed Request"
            scores = ["no", 0]

        # Update the dictionary with the new entries
        updated_dict = {"video_name": data_dict["video_name"], "Correctness": scores[0], "score": scores[1], "Q": question, "A": answer, "pred": pred, "question_id": data_dict.get("question_id"), "type": data_dict.get("type")}
        evaluated_results.append(updated_dict)

    return evaluated_results


# Factory into different aggregate
def activitynetqa_aggregate_score(results, args):
    yes_count = 0
    no_count = 0
    total_score = 0

    # Iterate over the results to count correctness and sum scores
    for result_dict in results:
        if "yes" in result_dict["Correctness"].lower():
            yes_count += 1
        elif "no" in result_dict["Correctness"].lower():
            no_count += 1

        total_score += int(result_dict["score"])

    # Calculate accuracy and average score
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    average_score = total_score / len(results) if results else 0
    eval_logger.info(f"Accuracy: {accuracy}")
    eval_logger.info(f"Average Score: {average_score}")
    return average_score


def activitynetqa_aggregate_accuracy(results, args):
    yes_count = 0
    no_count = 0
    total_score = 0

    # Iterate over the results to count correctness and sum scores
    for result_dict in results:
        if "yes" in result_dict["Correctness"].lower():
            yes_count += 1
        elif "no" in result_dict["Correctness"].lower():
            no_count += 1

        total_score += int(result_dict["score"])

    # Calculate accuracy and average score
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    average_score = total_score / len(results) if results else 0
    eval_logger.info(f"Accuracy: {accuracy}")
    eval_logger.info(f"Average Score: {average_score}")
    return accuracy * 100


