import random

from openai import OpenAI
import json
import itertools
import os
import csv
import re
import pandas as pd

# e.g., instruct_file: non_fqn_instruct; examples_file: non_fqn_example; user_input: API_text
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

# def write2file(dict,file_name):
#     df = pd.DataFrame(dict)
#     df.to_csv(file_name, header=False, index=False)

def read_API_file():
    with open('../../Source/API_List_Java.csv', encoding='utf-8') as f:
        return [row[0].lower() for row in csv.reader(f)]

class Module_1:
    def __init__(self):
        self.non_fqn_extraction_instruct = '../../prompt_instruct/java/parse_non-fqn'
        self.non_fqn_extraction_example = '../../prompt_example/java/parse_non-fqn.csv'
        self.fqn_inference_instruct = '../../prompt_instruct/java/infer_fqn'
        self.fqn_inference_example = '../../prompt_example/java/infer_fqn.csv'
        # non_fqn_input = "Natural language text: " + API_Text
        # non_fqn_message = load_messages(self.non_fqn_instruct,self.non_fqn_example,non_fqn_input)

    # AI-UNIT-1
    def Extract_Non_FQN(self,message):
        API_Key = ['Keys']
        random.shuffle(API_Key)
        API_Base = os.environ.get("OpenAI_API_Base", "https://api.openai.com/v1")
        API_Key = os.environ.get("Open_API_Key", API_Key[0])
        client = OpenAI(api_key=API_Key, base_url=API_Base)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content.strip()

    # AI-UNIT-2
    def Infer_FQN(self,message):
        API_Key = ['Keys']
        random.shuffle(API_Key)
        API_Base = os.environ.get("OpenAI_API_Base", "https://api.openai.com/v1")
        API_Key = os.environ.get("Open_API_Key", API_Key[0])
        client = OpenAI(api_key=API_Key, base_url=API_Base)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content.strip()

def execute_module_1(API_Text):
    API_pair_list = list()
    fqns_list = list()

    module_instance_1 = Module_1()
    module_instance_2 = Module_1()

    unit_1_input = "Natural language text: " + API_Text     # API_Text
    unit1_message = load_message(module_instance_1.non_fqn_extraction_instruct, module_instance_1.non_fqn_extraction_example, unit_1_input)
    unit1_output = module_instance_1.Extract_Non_FQN(unit1_message)     # non-fqns

    api_list = read_API_file()
    text_tokens = API_Text.replace("(","").replace(")","").split(", ")
    for element in text_tokens:
        if element.lower() in api_list:
            fqns_list.append(element)
            fqns_list = list(set(fqns_list))

    unit_2_input = unit_1_input + '\n' + unit1_output       # API_text + non-fqns
    if "None" not in unit1_output:
        unit2_message = load_message(module_instance_2.fqn_inference_instruct, module_instance_2.fqn_inference_example, unit_2_input)
        unit2_output = module_instance_2.Infer_FQN(unit2_message)       # fqns
        fqns_list.extend(unit2_output.split(": ")[1].split('; '))
        fqns_list = list(set(fqns_list))
        if len(fqns_list) >= 2:
            API_pair_list = list(itertools.combinations(fqns_list, 2)) # [(API_1, API_2),(API_3,API_4)]
    elif "None" in unit1_output and len(fqns_list)>1:
        unit2_output = "None"
        API_pair_list = list(itertools.combinations(fqns_list, 2))
    else:
        unit2_output = "None"

    return unit1_output, unit2_output, API_pair_list
