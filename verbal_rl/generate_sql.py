class Generate_SQL:
    def __init__(self, args):
        self.args = args
        self.model = self.args['model']
        self.temperature = self.args['temperature']
        self.max_tokens = self.args['max_tokens']
        self.prompt = None

        if self.args['load_model']:
            self.client, self.tokenizer = self.load_model()
        elif self.model in self.args['local_api']:
            self.headers = {"Content-Type": "application/json"}
        elif self.model in self.args['openai_api']:
            import openai
            openai.api_key = self.args['api_key']
            self.client = openai
        elif self.model in self.args['anthropic_api']:
            from anthropic import Anthropic
            self.client = Anthropic(
                api_key=self.args['api_key']
            )
        else:
            self.client = None
            
    def load_model(self):
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        os.environ['CUDA_VISIBLE_DEVICES'] = self.args['gpus']
        torch.cuda.manual_seed(self.args['seed'])
        torch.manual_seed(self.args['seed'])

        model = AutoModelForCausalLM.from_pretrained(
            self.args['model_path'],  
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            use_cache=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(self.args['model_path'])
        return model, tokenizer
    
    #From OmniSQL git train_and_evaluate/infer.py
    def parse_response(self, response, str_pattern='sql'):
        import re
        if str_pattern == 'sql':
            pattern = r"```sql\s*(.*?)\s*```"
        elif str_pattern == 'think':
            pattern = r"<think>\s*(.*?)\s*</think>"
        elif str_pattern == 'decision':
            pattern = r"<decision>\s*(.*?)\s*</decision>"
        elif str_pattern == 'scores':
            pattern = r"<scores>\s*(.*?)\s*</scores>"
   
        sql_blocks = re.findall(pattern, response, re.DOTALL)

        if sql_blocks:
            # Extract the last SQL query in the response text and remove extra whitespace characters
            last_sql = sql_blocks[-1].strip()
            return last_sql
        else:
            # print("No SQL blocks found.")
            return ""
   
    def prompt_sql(self, prompt=None):
        self.prompt = prompt
        return

    def prompt_decision(self, org_prompt, res_sql, res_think= None):
        org_part = """Instructions:
- Please use the minimum number of tokens required to provide a SQL statement and use sql functions for SQLite database.
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Please think through the steps of how to write the query with minimum number of tokens.

Output Format:
In your answer, please enclose the thinking block with less than 1536 tokens followed by a code block with the generated SQL code:
<think>
-- Your brief thinking
</think>
```sql
-- Your SQL query
```

Please DO NOT generate any explanation to the final SQL code solution."""

        rep_part = f"""THINKING: 
{res_think}

Instructions:
Based on understanding from DATABASE SCHEMA, DOCUMENTATIONS, and THINKING,
please identify whether the following SQL script can answer the input QUESTION or not:
SQL:
{res_sql}

If the input SQL script is CORRECT,
- Please provide the input SQL query as Your SQL query in the output.
- Please provide decision 'END' as Youe decision in the output.

If the input SQL script is NOT CORRECT, 
- Please use the minimum number of tokens required to provide a new SQL statement and use sql functions for SQLite database.
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Please think through the steps of how to write the query with minimum number of tokens.
- Please provide decision 'CONTINUE' as Youe decision in the output.

Output Format:
In your answer, please enclose the thinking block with less than 1536 tokens followed by a code block with the generated SQL code:
<think>
-- Your brief thinking
</think>
```sql
-- Please provide the input SQL query as Your SQL query
```
<decision>
-- Your decision: If the input SQL script is CORRECT, Please provide decision 'END'; If the input SQL script is NOT CORRECT, Please provide decision 'CONTINUE'
</decision>

Please DO NOT generate any explanation to the final SQL code solution and decision."""            

        self.prompt = org_prompt.replace(org_part, rep_part)
        return
    
    def prompt_scores(self, org_prompt, all_sqls): 
        org_part1 = """You are a powerful text-to-SQL model. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question."""
        rep_part1 = f"""You are a scoring machine to make scores for {len(all_sqls)} SQL scripts, based on understanding from input DATABASE SCHEMA, CONTEXT and QUESTION."""
       
        org_part2 = """Instructions:
- Please use the minimum number of tokens required to provide a SQL statement and use sql functions for SQLite database.
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Please think through the steps of how to write the query with minimum number of tokens.

Output Format:
In your answer, please enclose the thinking block with less than 1536 tokens followed by a code block with the generated SQL code:
<think>
-- Your brief thinking
</think>
```sql
-- Your SQL query
```

Please DO NOT generate any explanation to the final SQL code solution."""

        sqls = ""
        scores = ""
        for i in range(len(all_sqls)):
            sqls += f"SQL{i}\n"
            sqls += all_sqls[i] + '\n\n'
            scores += f"SQL{i}: --score between 0 and 1: up to second digit"
            if i < len(all_sqls) - 1:
                scores += '\n'

        rep_part2 = f"""Instructions for scoring SQL scripts:
- Based on your understanding from input DATABASE SCHEMA, CONTEXT and QUESTION, please give score for each SQL script.
- Please compare all SQL scripts and give high score for SQL script that fully answers the input QUESTION.
- Please think through the steps with minimum number of tokens and the score should be between 0 and 1.

{sqls}

Output Format:
In your answer, please enclose the thinking block with less than 1536 tokens followed by a code block with scores for each SQL script:
<think>
-- Your brief thinking
</think>
<scores>
{scores}
</scores>

Please DO NOT generate any explanation to the final scores."""            

        self.prompt = org_prompt.replace(org_part1, rep_part1)
        self.prompt = self.prompt.replace(org_part2, rep_part2)
        return
        
    def gen_output(self):
        if self.args['load_model']:
            self.client.to(self.args['device'])
            inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.args['device'])
            resp = self.client.generate(**inputs, max_length=self.max_tokens)
            output = self.tokenizer.decode(resp[0], skip_special_tokens=True)
        elif self.model in self.args['local_api']:

            import requests
            import json
            payload = {
                "model" : self.model, 
                "prompt": self.prompt,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            resp = requests.post(
                self.args['api_url'], 
                headers=self.headers, 
                json=payload).json()

            if 'choices' in resp:
                output = resp['choices'][0]['text'].strip()
            else:
                output = ''
        else:
            if self.model in self.args['anthropic_api']:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": self.prompt}]
                )
                output = resp.content[0].text.strip()            
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': self.prompt}]
                )
                output = resp.choices[0].message.content.strip()
        return output