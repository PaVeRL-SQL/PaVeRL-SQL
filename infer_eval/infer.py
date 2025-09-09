import os
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN']='1'

import re
import pandas as pd

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

seed=2025
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', dest='mode', type=str, default='greedy')
parser.add_argument('--model', dest='model', type=str, default='')
parser.add_argument('--dpath', dest='dp', type=str, default='')
parser.add_argument('--outputpath', dest='outputs', type=str, default='')
parser.add_argument('--nmaj', dest='n', type=int, default=8)
args = parser.parse_args()
dpath=args.dp
model_name = args.model
mode = args.mode
output_path = args.outputs
n_maj = args.n

if not os.path.isdir(output_path):
    os.makedirs(output_path)

#Qwen2.5 stop tokens
stop_token_ids = [151645]
errflag=0
inferdf=pd.read_csv(os.path.join(dpath, 'test.csv'))

if mode=='greedy':
    sampling_params = SamplingParams(
        temperature = 0, 
        max_tokens = 2048,
        n = 1,
        stop_token_ids = stop_token_ids
    )
    llm=LLM(
            model = model_name,
            dtype = "float16", 
            tensor_parallel_size = 2,
            max_model_len = 64000,
            gpu_memory_utilization = 0.8,
            swap_space = 8,
            enforce_eager = True,
            disable_custom_all_reduce = True,
            trust_remote_code = True
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
elif mode=='majority':
    sampling_params = SamplingParams(
        temperature = 0.8, 
        max_tokens = 2048,
        n = n_maj,
        stop_token_ids = stop_token_ids
    )
    llm=LLM(
            model = model_name,
            dtype = "float16", 
            tensor_parallel_size = 2,
            max_model_len = 64000,
            gpu_memory_utilization = 0.8,
            swap_space = 8,
            enforce_eager = True,
            disable_custom_all_reduce = True,
            trust_remote_code = True
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
else:
    errflag=1
    print("Mode Error")

#From OmniSQL git train_and_evaluate/infer.py
def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"
    
    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        # print("No SQL blocks found.")
        return ""

if errflag==0:
    chat_prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": inferdf.loc[i, 'promptin']}],
            add_generation_prompt = True, tokenize = False
        ) for i in range(len(inferdf))]
    outputs = llm.generate(chat_prompts, sampling_params)
    result=[]
    predsqls=[]
    for output in outputs:
        responses= [o.text for o in output.outputs]
        sqls=[parse_response(response) for response in responses]
        result.append(responses)
        predsqls.append(sqls)

    resdf=inferdf[['testid','dbname','question','evidence','goldsql','goldcsv']].copy()
    resdf['genout']= result
    resdf['pred_sqls']=predsqls

    md=[x for x in model_name.split('/') if x][-1]
    d=[x for x in dpath.split('/') if x][-1]
    fname=f'{output_path}/{md}_{d}_{mode}_{n_maj}.json'
    resdf.to_json(fname, orient='records',index=False)