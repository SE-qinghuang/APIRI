import ast
import os
import random
from openai import OpenAI
import json
import csv
import pandas as pd

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

def form_know(know1, know2):
    return know1 + '\n' + know2

def combine_knowledge(know1,know2):
    k_1 = form_know(know1["km_usage"], know2["km_usage"])
    k_2 = form_know(know1["km_characteristic"], know2["km_characteristic"])
    k_3_1 = form_know(know1["km_replace_1"], know2["km_replace_2"])
    k_3_2 = form_know(know2["km_replace_1"], know1["km_replace_2"])
    k_4 = form_know(know1["km_efficiency"], know2["km_efficiency"])
    k_5 = form_know(know1["km_logic"], know2["km_logic"])
    k_6 = form_know(know1["km_task"], know2["km_task"])
    k_7 = form_know(know1["km_conversion"], know2["km_conversion"])
    pair_Knowledge = (k_1, k_2, k_3_1, k_3_2, k_4, k_5, k_6,k_7)
    return pair_Knowledge

class Module_2:
    def __init__(self, API):
        self.API = API
        base_instruct_path = "../../prompt_instruct/java/{key}"
        base_example_path = "../../prompt_example/java/{key}.csv"
        keys = [
            'km_usage','km_characteristic','km_replace_1','km_replace_2','km_efficiency','km_logic','km_task','km_conversion',
        ]
        self.instructs = {key: base_instruct_path.format(key=key) for key in keys}
        self.examples = {key: base_example_path.format(key=key) for key in keys}
        self.questions = {
            'km_usage': "Q: What is the primary usage of {}?".format(API),
            'km_characteristic': "Q: What are the characteristics of {}?".format(API),
            'km_replace_1': "Q: When should i use {}?".format(API),
            'km_replace_2': "Q: When should i not use {}?".format(API),
            'km_efficiency': "Q: What is the performance of {}?".format(API),
            'km_logic': "Q: What should i do before and after using {}?".format(API),
            'km_task': "Q: What tasks can {} accomplish?".format(API),
            'km_conversion': "Q: What data types can {} be converted to?".format(API),
        }

    def Mine_API_Knowledge(self,message):
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
        return response.choices[0].message.content.strip().replace("A: ","")

# input: API_1, API_2, Model_Name; output: API_Knowledge
def execute_module_2(API_pairs):
    pair_knowledge = list()
    unique_apis = set(api for pair in API_pairs for api in pair)
    api_knowledge = {}
    for api in unique_apis:
        module_instance = Module_2(api)
        keys = ['km_usage','km_characteristic','km_replace_1','km_replace_2','km_efficiency','km_logic','km_task','km_conversion',]
        messages = {}
        knowledge = {}
        for key in keys:
            messages[key] = load_message(module_instance.instructs[key],module_instance.examples[key],module_instance.questions[key])
            knowledge[key] = module_instance.Mine_API_Knowledge(messages[key])
        api_knowledge[api] = knowledge

    for pair in API_pairs:
        # pair (api_1,api_2)
        knowledge_dict = {
            "API_pair": pair,
            "API_Knowledge": combine_knowledge(api_knowledge[pair[0]],api_knowledge[pair[1]])
        }
        pair_knowledge.append(knowledge_dict)

    return pair_knowledge
