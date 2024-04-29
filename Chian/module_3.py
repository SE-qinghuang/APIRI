import ast
import csv
import os
import re
from collections import Counter
from openai import OpenAI
import json
import pandas as pd
import random

def load_message(instruct_file, examples_file, user_input):
    with open(instruct_file, 'r', encoding='utf-8') as system_file:
        instruct_content = system_file.read()
    message = [{"role": "system", "content": instruct_content}]
    conversation_df = pd.read_csv(examples_file)
    for _, row in conversation_df.iterrows():
        message.append({"role": "user", "content": row["user"]})
        message.append({"role": "assistant", "content": row["assistant"]})
    message.append({"role": "user", "content": user_input})
    return message


def modify_lists(input_list):
    if input_list[2] == "no" and input_list[3] == "no":
        input_list[2] = "no"
    else:
        input_list[2] = "yes"
    del input_list[3]

    return input_list

def aggregation(res_1_list, res_2_list, res_3_list):
    final_answer = []
    for val1, val2, val3 in zip(res_1_list, res_2_list, res_3_list):
        values = [val1, val2, val3]
        most_frequent = Counter(values).most_common(1)[0][0]
        final_answer.append(values + [most_frequent])
    return final_answer


def load_knowledge(API_Knowledge):
    k4r1 = API_Knowledge[0]
    k4r2 = '\n'.join(API_Knowledge[0], API_Knowledge[1])
    k4r3_1 = API_Knowledge[2]
    k4r3_2 = API_Knowledge[3]
    k4r4 = '\n'.join(API_Knowledge[0], API_Knowledge[4])
    k4r5 = '\n'.join(API_Knowledge[0], API_Knowledge[5])
    k4r6 = API_Knowledge[6]
    k4r7 = API_Knowledge[7]
    return (k4r1,k4r2,k4r3_1,k4r3_2,k4r4,k4r5,k4r6,k4r7)
class Module_3:
    def __init__(self, API_1,API_2,API_Knowledge):
        self.API_1 = API_1
        self.API_2 = API_2
        self.API_Knowledge = API_Knowledge
        # Define the common attributes for both instructs and examples
        categories = {
            'yn_instructs': ('yn', 'instruct', ''),
            'yn_examples': ('yn', 'example', '.csv'),
            'tf_instructs': ('tf', 'instruct', ''),
            'tf_examples': ('tf', 'example', '.csv')
        }
        types = ['similar', 'difference', 'replace', 'efficiency', 'logic', 'collaboration', 'conversion']
        # Dynamically generate the dictionaries
        for category, (prefix, category_type, extension) in categories.items():
            setattr(self, category, {
                f'{prefix}_{type}': f"../../prompt_{category_type}/java/{prefix}_{type}{extension}"
                for type in types
            })

        self.mul_instructs = "../../prompt_instruct/java/mul_all"
        self.mul_examples = "../../prompt_example/java/mul_all.csv"

        know_chunk = load_knowledge(self.API_Knowledge)

        question_template = "API_Knowledge:\n{}\n{}"
        yn_content = {
            "yn_similar": "Q: Based on the API knowledge above, do {API_1} and {API_2} have similar usage?",
            "yn_difference": "Q: Based on the API knowledge above, do {API_1} and {API_2} have similar usage but different behavior?",
            "yn_replace_1": "Q: Can {API_1} be used in the above unavailable scenarios of {API_2}?",
            "yn_replace_2": "Q: Can {API_2} be used in the above unavailable scenarios of {API_1}?",
            "yn_efficiency": "Q: Based on the API knowledge above, do {API_1} and {API_2} have similar usage and efficiency comparison?",
            "yn_logic": "Q: Based on the API knowledge above, is there a logical order when using {API_1} and {API_2}?",
            "yn_collaboration": "Q: Based on the API knowledge above, is there a task scenario that requires {API_1} and {API_2} to cooperate?",
            "yn_conversion": "Q: Based on the knowledge above, can the data types of {API_1} and {API_2} be converted to each other?",
        }
        tf_content = {
        "tf_similar": "Claim: Based on the API Knowledge above, {API_1} and {API_2} have similar usage.",
        "tf_difference": "Claim: Based on the API Knowledge above, {API_1} and {API_2} have similar usage but different behavior.",
        "tf_replace_1": "Claim: {API_1} can be used in the above unavailable scenarios of {API_2}.",
        "tf_replace_2": "Claim: {API_2} can be used in the above unavailable scenarios of {API_1}.",
        "tf_efficiency": "Claim: Based on the API knowledge above, {API_1} and {API_2} have similar usage and efficiency comparison.",
        "tf_logic": "Claim: Based on the API knowledge above, there is a logical order when using {API_1} and {API_2}.",
        "tf_collaboration": "Claim: Based on the API knowledge above, there is a task scenario that requires {API_1} and {API_2} to cooperate.",
        "tf_conversion": "Claim: Based on the knowledge above, the data types of {API_1} and {API_2} can be converted to each other."
        }
        self.yn_questions = {
            key: question_template.format(know_chunk[i],
                yn_content[key].format(API_1=API_1, API_2=API_2)
            ) for i, (key, question) in enumerate(yn_content.items())
        }
        self.tf_questions = {
            key: question_template.format(
                know_chunk[i],
                tf_content[key].format(API_1=API_1, API_2=API_2)
            ) for i, (key, question) in enumerate(tf_content.items())
        }
    def qa_decider(self, message):
        API_Key = ['Keys']
        random.shuffle(API_Key)
        API_Base = os.environ.get("OpenAI_API_Base", "https://api.openai.com/v1")
        API_Key = os.environ.get("Open_API_Key", API_Key[0])
        client = OpenAI(api_key=API_Key, base_url=API_Base)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0,
            max_tokens=8,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        # print("yn", response.choices[0].message.content.strip().lower())
        if "yes" in response.choices[0].message.content.strip().lower():
            return "yes"
        else:
            return "no"

    def tf_decider(self, message):
        API_Key = ['Keys']
        random.shuffle(API_Key)
        API_Base = os.environ.get("OpenAI_API_Base", "https://api.openai.com/v1")
        API_Key = os.environ.get("Open_API_Key", API_Key[0])
        client = OpenAI(api_key=API_Key, base_url=API_Base)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0,
            max_tokens=8,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        # print("tf",response.choices[0].message.content.strip().lower())
        if "incorrect" in response.choices[0].message.content.strip().lower():
            return "no"
        else:
            return "yes"

    def mul_decider(self, message):
        API_Key = ['Keys']
        random.shuffle(API_Key)
        API_Base = os.environ.get("OpenAI_API_Base", "https://api.openai.com/v1")
        API_Key = os.environ.get("Open_API_Key", API_Key[0])
        client = OpenAI(api_key=API_Key, base_url=API_Base)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0,
            max_tokens=128,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        if "unknow" not in response.choices[0].message.content.strip().lower():
            convert_answer = response.choices[0].message.content.strip().lower().split(": ")[1]
            mul_result_list =  convert_answer.split(', ')
            categories = ["function similarity", "behavior difference", "function replace", "efficiency comparison",
                          "logic constraint", "function collaboration","conversion"]
            output_list = ['no'] * 7
            for input_category in mul_result_list:
                if input_category in categories:
                    index = categories.index(input_category)
                    output_list[index] = 'yes'
        else:
            output_list = ['no'] * 7
        return output_list

def execute_module_3(API_pairs_knowledge):
    out = []
    for pandk in API_pairs_knowledge:
        task_3 = {"API_1":None,"API_2":None,"Rels":None}
        API_1 = pandk["API_pair"][0]
        API_2 = pandk["API_pair"][1]
        task_3["API_1"] = API_1
        task_3["API_2"] = API_2
        API_Knowledge = pandk["API_Knowledge"]
        module_instance = Module_3(API_1, API_2, API_Knowledge)
        question_types = [
            ('yn', ['similar', 'difference', 'replace_1', 'replace_2', 'efficiency', 'logic', 'collaboration', 'conversion']),
            ('tf', ['similar', 'difference', 'replace_1', 'replace_2', 'efficiency', 'logic', 'collaboration', 'conversion'])
        ]
        yn_result = {}
        tf_result = {}
        for question_prefix, question_keys in question_types:
            for key_suffix in question_keys:
                key = f'{question_prefix}_{key_suffix}'
                instructs_key = f'{question_prefix}_instructs'
                examples_key = f'{question_prefix}_examples'
                questions_key = f'{question_prefix}_questions'
                if key_suffix != "replace_1" and key_suffix != "replace_2":
                    input_message = load_message(getattr(module_instance, instructs_key)[key],
                                                 getattr(module_instance, examples_key)[key],
                                                 getattr(module_instance, questions_key)[key])
                else:
                    file_path = f'{question_prefix}_replace'
                    input_message = load_message(getattr(module_instance, instructs_key)[file_path],
                                                 getattr(module_instance, examples_key)[file_path],
                                                 getattr(module_instance, questions_key)[key])

                # Determine and call the appropriate decision function
                if question_prefix == 'yn':
                    yn_result[key_suffix] = module_instance.qa_decider(input_message)
                else:  # 'tf'
                    tf_result[key_suffix] = module_instance.tf_decider(input_message)

        yn_res = modify_lists(list(yn_result.values()))
        tf_res = modify_lists(list(tf_result.values()))

        mul_input = "API Knowledge: \n" + "\n".join(API_Knowledge)
        mul_message = load_message(module_instance.mul_instructs,module_instance.mul_examples,mul_input)
        mul_res = module_instance.mul_decider(mul_message)

        final_answer = aggregation(yn_res,tf_res,mul_res)
        task_3["Rels"] = final_answer
        out.append(task_3)

    return out
