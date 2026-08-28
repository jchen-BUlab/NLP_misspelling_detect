#****************************************
# MIT License
# Copyright (c) 2026 Jiayu Lu, Jinying Chen
#  
# author(s): Jiayu Lu, Jinying Chen, Boston University Chobanian & Avedisian School of Medicine
# date: 2026-8-15
# ver: 1.0
# 
# This code was written to support model evaluation for the 2026 paper published 
# in JMIR Medical Informatics. 
# The code is for research use only, and is provided as it is.
# 
# See LICENSE in the project root for license terms.

import argparse
import itertools
import os
import re
import socket
import time

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm


PROMPT_TEMPLATES = {
    "baseline0": {
        "user": (
            "For each of the following {count} lines, classify the line as '0' if it is a "
            "correctly spelled medication name or '1' otherwise. "
            "Return only 0 or 1 for each line, and nothing else. "
            "The response should contain {count} classifications and split by space.\n"
            "\n{lines}\n"
        ),
        "single_user": (
            "Classify this medication name as '0' if it is a correctly spelled medication name "
            "or '1' otherwise. Return only 0 or 1, and nothing else.\n\n{name}\n"
        ),
    },
    "baseline": {
        "system": "You are a medication name spelling checker.",
        "user": (
            "For each of the following {count} lines, classify the line as '0' if it is a "
            "correctly spelled medication name or '1' otherwise. Return only 0 or 1 for each "
            "line, and nothing else. The response should contain {count} classifications "
            "separated by spaces.\n\n{lines}\n"
        ),
        "single_user": (
            "Classify this medication name as '0' if it is a correctly spelled medication name "
            "or '1' otherwise. Return exactly one label and nothing else.\n\n{name}\n"
        ),
    },
    "strict": {
        "system": (
            "You are a careful medication-name spelling auditor. Judge each line independently "
            "and do not explain your reasoning."
        ),
        "user": (
            "Task: classify each line with a binary label.\n"
            "Label definitions:\n"
            "0 = clearly a correctly spelled medication name.\n"
            "1 = misspelled medication name, nonstandard spelling, non-medication term, noisy "
            "text, or any case where you are not confident it is a correctly spelled medication "
            "name.\n"
            "Rules:\n"
            "- Brand and generic medication names are valid when spelled correctly.\n"
            "- Do not autocorrect or rewrite the input.\n"
            "- If a line contains a typo, return 1.\n"
            "- If uncertain, return 1.\n"
            "- Output exactly {count} labels separated by single spaces.\n"
            "- Output only the labels.\n\n"
            "Lines:\n{lines}\n"
        ),
        "single_user": (
            "Task: classify this medication name with a binary label.\n"
            "Label definitions:\n"
            "0 = clearly a correctly spelled medication name.\n"
            "1 = misspelled medication name, nonstandard spelling, non-medication term, noisy "
            "text, or any case where you are not confident it is a correctly spelled medication "
            "name.\n"
            "Rules:\n"
            "- Brand and generic medication names are valid when spelled correctly.\n"
            "- Do not autocorrect or rewrite the input.\n"
            "- If the name contains a typo, return 1.\n"
            "- If uncertain, return 1.\n"
            "- Output exactly one label: 0 or 1.\n"
            "- Output only the label.\n\n"
            "Medication name:\n{name}\n"
        ),
    },
    "few_shot": {
        "system": (
            "You are a careful medication-name spelling auditor. Judge spelling only and return "
            "machine-readable labels."
        ),
        "user": (
            "Classify each input line.\n"
            "0 = correctly spelled medication name.\n"
            "1 = misspelled medication name, nonstandard spelling, non-medication term, noisy "
            "text, or uncertain case.\n\n"
            "Examples:\n"
            "metformin -> 0\n"
            "amoxicillin -> 0\n"
            "metfornin -> 1\n"
            "amoxcillin -> 1\n"
            "headache -> 1\n\n"
            "Now classify the next {count} lines. Output exactly {count} labels separated by "
            "single spaces. Output only the labels.\n\n"
            "{lines}\n"
        ),
        "single_user": (
            "Classify this input.\n"
            "0 = correctly spelled medication name.\n"
            "1 = misspelled medication name, nonstandard spelling, non-medication term, noisy "
            "text, or uncertain case.\n\n"
            "Examples:\n"
            "metformin -> 0\n"
            "amoxicillin -> 0\n"
            "metfornin -> 1\n"
            "amoxcillin -> 1\n"
            "headache -> 1\n\n"
            "Now classify this medication name. Output exactly one label: 0 or 1. "
            "Output only the label.\n\n"
            "{name}\n"
        ),
    },
}


def build_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    return openai.OpenAI(
        organization=os.getenv("OPENAI_ORG_ID", "###"),
        project=os.getenv("OPENAI_PROJECT_ID", "###"),
        api_key=os.getenv("OPENAI_API_KEY", "###"),
    )


client = build_client()

def parse_float_list(raw_value):
    return [float(value.strip()) for value in raw_value.split(",") if value.strip()]


def sanitize_config_name(*parts):
    raw_name = "_".join(str(part) for part in parts)
    return raw_name.replace(".", "p").replace("-", "neg").replace("/", "_")


def build_prompt(prompt_style, batch):
    template = PROMPT_TEMPLATES[prompt_style]
    system_message = template.get("system")
    if len(batch) == 1:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": template["single_user"].format(name=batch[0])})
        return messages

    lines = "\n".join(batch)
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": template["user"].format(count=len(batch), lines=lines)})
    return messages


def parse_binary_predictions(output, expected_count):
    split_predictions = output.strip().split()
    if len(split_predictions) == expected_count and set(split_predictions).issubset({"0", "1"}):
        return [int(value) for value in split_predictions]

    fallback_predictions = re.findall(r"\b[01]\b", output)
    if len(fallback_predictions) >= expected_count:
        return [int(value) for value in fallback_predictions[:expected_count]]

    raise ValueError(
        f"Expected {expected_count} binary predictions, received {len(fallback_predictions)} from output: {output!r}"
    )


def _candidate_probability(top_logprobs, label):
    target = str(label)
    for candidate in top_logprobs or []:
        if candidate.token.strip() == target:
            return float(np.exp(candidate.logprob))
    return None


def extract_label_one_probabilities(logprob_content, predictions):
    probabilities = []
    prediction_index = 0
    for token_info in logprob_content or []:
        stripped = token_info.token.strip()
        if stripped not in {"0", "1"}:
            continue

        prob_one = _candidate_probability(token_info.top_logprobs, label=1)
        if prob_one is None:
            prob_zero = _candidate_probability(token_info.top_logprobs, label=0)
            if prob_zero is not None:
                prob_one = max(0.0, min(1.0, 1.0 - prob_zero))
            else:
                prob_one = float(predictions[prediction_index])

        probabilities.append(prob_one)
        prediction_index += 1
        if prediction_index == len(predictions):
            break

    if len(probabilities) != len(predictions):
        return []
    return probabilities


def get_completion_kwargs(model, temperature, top_p, use_logprobs):
    kwargs = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
    }
    if use_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5
    return kwargs


def check_spelling_batch(
    model,
    words,
    prompt_style="strict",
    temperature=0.0,
    top_p=1.0,
    batch_size=50,
    max_retries=3,
    use_logprobs=True,
):
    predicts = []
    predicts_logits = []
    total_api_time = 0.0
    messages_description = (
        f"Scoring {len(words)} items with model={model}, prompt_style={prompt_style}, "
        f"temperature={temperature}, top_p={top_p}"
    )
    for batch_start in tqdm(range(0, len(words), batch_size), desc=messages_description):
        batch = words[batch_start:batch_start + batch_size]
        messages = build_prompt(prompt_style, batch)

        for retry in range(max_retries):
            try:
                api_start_time = time.time()
                response = client.chat.completions.create(
                    messages=messages,
                    **get_completion_kwargs(model, temperature, top_p, use_logprobs),
                )
                total_api_time += time.time() - api_start_time
                output = response.choices[0].message.content or ""
                parsed_predictions = parse_binary_predictions(output, expected_count=len(batch))
                predicts.extend(parsed_predictions)

                if use_logprobs:
                    batch_probabilities = extract_label_one_probabilities(
                        getattr(response.choices[0].logprobs, "content", None),
                        parsed_predictions,
                    )
                    if len(batch_probabilities) == len(parsed_predictions):
                        predicts_logits.extend(batch_probabilities)
                    else:
                        predicts_logits.extend([float(value) for value in parsed_predictions])
                break
            except Exception as error:
                if retry == max_retries - 1:
                    print(f"Error processing batch {batch_start // batch_size + 1}: {error}")
                    fallback_predictions = [1] * len(batch)
                    predicts.extend(fallback_predictions)
                    if use_logprobs:
                        predicts_logits.extend([1.0] * len(batch))
                    break
                print(f"Retrying batch {batch_start // batch_size + 1} after error: {error}")
                time.sleep(5)

    return predicts, predicts_logits, total_api_time


def load_dataset(input_folder, filename):
    dataset_path = os.path.join(input_folder, filename)
    df = pd.read_csv(dataset_path)
    required_columns = {"input", "target"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{dataset_path} is missing required column(s): {missing_text}")
    return df, dataset_path


def run_api_on_dataset(
    df,
    model,
    prompt_style,
    temperature,
    top_p,
    batch_size,
    max_retries,
    use_logprobs,
):
    run_df = df.copy()
    api_predicts, api_predicts_logits, total_api_time = check_spelling_batch(
        model=model,
        words=run_df["input"].tolist(),
        prompt_style=prompt_style,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        max_retries=max_retries,
        use_logprobs=use_logprobs,
    )

    config_name = sanitize_config_name(model, prompt_style, temperature, top_p)
    prediction_column = f"{config_name}_predict"
    probability_column = f"{config_name}_prob"
    run_df[prediction_column] = api_predicts
    if use_logprobs:
        run_df[probability_column] = api_predicts_logits
    process_speed = total_api_time / len(run_df) if len(run_df) else 0.0
    running_env = socket.gethostname()
    run_df["process_time"] = total_api_time
    run_df["process_speed"] = process_speed
    run_df["running_env"] = running_env

    return run_df, prediction_column, probability_column if use_logprobs else None, total_api_time, process_speed, running_env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--input_folder", default="analysis_gpt_api_expts")
    parser.add_argument("--test_filename", default="gpt_api_test_set.csv")
    parser.add_argument("--tuning_filename", default="gpt_api_tuning_set.csv")
    parser.add_argument("--mode", choices=["tuning", "testing"], default="tuning")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--output_folder", default="analysis_gpt_api_results")
    parser.add_argument("--prompt_styles", default="strict,few_shot")
    parser.add_argument("--temperatures", default="0.0,0.2,0.7")
    parser.add_argument("--top_ps", default="1.0,0.9")
    parser.add_argument("--disable_logprobs", action="store_true")
    return parser.parse_args()


def get_mode_output_folder(output_folder, mode, model):
    if mode == "tuning":
        return os.path.join(output_folder, mode, sanitize_config_name(model))
    return os.path.join(output_folder, mode)


def main():
    args = parse_args()
    mode_to_filename = {
        "testing": args.test_filename,
        "tuning": args.tuning_filename,
    }
    mode_output_folder = get_mode_output_folder(args.output_folder, args.mode, args.model)
    os.makedirs(mode_output_folder, exist_ok=True)

    dataset_filename = mode_to_filename[args.mode]
    base_df, dataset_path = load_dataset(args.input_folder, dataset_filename)

    prompt_styles = [style.strip() for style in args.prompt_styles.split(",") if style.strip()]
    temperatures = parse_float_list(args.temperatures)
    top_ps = parse_float_list(args.top_ps)

    run_rows = []
    for prompt_style, temperature, top_p in itertools.product(prompt_styles, temperatures, top_ps):
        experiment_name = sanitize_config_name(args.model, prompt_style, temperature, top_p)
        print(f"\nRunning {experiment_name}")
        df, prediction_column, probability_column, total_api_time, process_speed, running_env = run_api_on_dataset(
            df=base_df,
            model=args.model,
            prompt_style=prompt_style,
            temperature=temperature,
            top_p=top_p,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            use_logprobs=not args.disable_logprobs,
        )

        output_path = os.path.join(mode_output_folder, f"{experiment_name}.csv")
        df.to_csv(output_path, index=False)

        run_rows.append(
            {
                "experiment": experiment_name,
                "mode": args.mode,
                "dataset_path": dataset_path,
                "model": args.model,
                "prompt_style": prompt_style,
                "temperature": temperature,
                "top_p": top_p,
                "prediction_column": prediction_column,
                "probability_column": probability_column or "",
                "process_time": total_api_time,
                "process_speed": process_speed,
                "running_env": running_env,
                "output_path": output_path,
            }
        )

    runs_df = pd.DataFrame(run_rows)
    runs_path = os.path.join(mode_output_folder, "openai_prompt_sweep_runs.csv")
    if os.path.exists(runs_path):
        existing_runs_df = pd.read_csv(runs_path)
        runs_df = pd.concat([existing_runs_df, runs_df], ignore_index=True)
        runs_df = runs_df.drop_duplicates(
            subset=[
                "experiment",
                "mode",
                "dataset_path",
                "model",
                "prompt_style",
                "temperature",
                "top_p",
                "prediction_column",
                "probability_column",
                "output_path",
            ],
            keep="last",
        )
    runs_df.to_csv(runs_path, index=False)


if __name__ == "__main__":
    main()
