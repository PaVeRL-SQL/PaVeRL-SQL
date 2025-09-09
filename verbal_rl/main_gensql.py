import os
import json
import argparse
import random
import sqlite3
import numpy as np
from data import Read_Data
from generate_sql import Generate_SQL
from execute_sql import QueryThread
from time import gmtime, strftime
import time

def get_args(path):
    with open(path, 'r') as f:
        args = json.load(f)
        
    if not os.path.isdir(args['save_path']):
        os.makedirs(args['save_path'])
    
    if not args['continue']:
        now = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        
        if os.path.isdir(os.path.join(args['save_path'], f"{args['modelname']}_{args['dataname']}_{now}")):
            os.rename(os.path.join(args['save_path'], f"{args['modelname']}_{args['dataname']}_{now}"), os.path.join(args['save_path'], f"{args['modelname']}_{args['dataname']}_{now}_old"))
        os.makedirs(os.path.join(args['save_path'], f"{args['modelname']}_{args['dataname']}_{now}"))

        with open(os.path.join(args['save_path'], f"{args['modelname']}_{args['dataname']}_{now}/info.json"), 'w') as f:
            json.dump(args, f)

        args['save_json'] = os.path.join(args['save_path'], f"{args['modelname']}_{args['dataname']}_{now}/results.json")

    else:
        with open(os.path.join(args['cont_path'], 'info.json'), 'w') as f:
            json.dump(args, f)

        args['save_json'] = os.path.join(args['cont_path'], 'results.json')

    return args

def main(args, all_qs, all_prompts, all_dbs):
    gen_sql = Generate_SQL(args)
    
    res = {}
    if args['continue']:
        with open(args['save_json'], 'r') as f:
            res = json.load(f)
    saved_ids = set(res.keys())

    for q_id in all_qs:
        if str(q_id) in saved_ids:
            print(f"{q_id} has been done.")
            continue
        
        call_api = 0
        question = all_qs[q_id]
        db = all_dbs[q_id]
        org_prompt = all_prompts[q_id]

        res[q_id] = {
            "question": question,
            "sql": '',
            "db": db,
            "max_score": None,
            "call_api": 0,
            "gen_sql_time": None,
            "scorre_time": None
        }

        print(q_id, 'Start generate SQL.')
        count = 1
        all_sqls = []
        start_time = time.time()
        while len(all_sqls) < args['max_num_sql'] and count <= args['max_round'] * args['max_num_sql']:
            print(f'{len(all_sqls)} SQL sripts ready.')
            gen_sql.prompt_sql(org_prompt)

            subcount = 1
            while subcount <= args['max_round']:
                output1 = gen_sql.gen_output()
                res_sql = gen_sql.parse_response(output1, str_pattern='sql')
                call_api += 1

                if 'select' in res_sql.lower() and '-- Your SQL query' not in res_sql[:17] and len(res_sql) > len('SELECT * FROM *'):
                    subcount = args['max_round'] + 1
                subcount += 1

            run_sql = QueryThread(res_sql, db)
            run_sql.start()
            run_sql.join(args['timeout'])
            
            print(q_id, 'End Run SQL.', run_sql.is_alive(), run_sql.exception)
            
            if not run_sql.is_alive():
                if run_sql.exception == None:
                    if run_sql.result.shape != (0, 0):
                        all_sqls.append(res_sql)
                        
            count += 1
        
        gen_sql_time = time.time() - start_time
        print(q_id, 'End generate SQL.', gen_sql_time)
        res[q_id]["gen_sql_time"] = gen_sql_time

        if len(all_sqls) == 1:
            res[q_id]["sql"] = all_sqls[0]
        elif len(all_sqls) > 1:
            print(q_id, 'Start scores.')
            start_time = time.time()
            gen_sql.prompt_scores(org_prompt, all_sqls)

            res_scores = [0.0 for _ in range(len(all_sqls))]
            subcount = 1
            while subcount <= args['max_round']:
                output2 = gen_sql.gen_output()
                output_scores = gen_sql.parse_response(output2, str_pattern='scores')
                call_api += 1
                
                temp_scores = output_scores.split('\n')
                if 'sql0:' in output_scores.lower() and len(all_sqls) == len(temp_scores):
                    for i in range(len(all_sqls)):
                        one_score = temp_scores[i]
                        try:
                            res_scores[i] += float(one_score.split(': ')[-1])
                        except:
                            res_scores[i] += 0.0
                    
                    subcount += 1

            res_scores = [a / args['max_round'] for a in res_scores]
            print(res_scores)
            score_time = time.time() - start_time
            print(q_id, 'End scores.', score_time)
            res[q_id]["score_time"] = score_time

            max_score = max(res_scores)
            targets = []
            for i in range(len(res_scores)):
                if max_score == res_scores[i]:
                    targets.append(i)
            res[q_id]["max_score"] = max_score
            print(targets)

            if len(targets) == 1:
                res[q_id]["sql"] = all_sqls[0]
            else:
                try:
                    res[q_id]["sql"] = all_sqls[random.choice(targets)]
                except:
                    res[q_id]["sql"] = ''

        res[q_id]["call_api"] = call_api
        new_res = {}
        for k, v in res.items():
            if isinstance(k, np.int64):
                new_res[int(k)] = v
            else:
                new_res[k] = v
        
        res = new_res
        with open(args['save_json'], 'w') as f:
            json.dump(res, f)
        print(q_id, 'End Save.')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str)
    args_path = parser.parse_args()

    args = get_args(args_path.path)
    all_qs, all_prompts, all_dbs = Read_Data().run(args)
    print(len(all_qs), len(all_prompts), len(all_dbs))
    main(args, all_qs, all_prompts, all_dbs)