import os
from tqdm import tqdm
import pandas as pd
import json
import ast
import argparse
from copy import deepcopy
import threading
import sqlite3
from collections import Counter
from func_timeout import func_timeout, FunctionTimedOut
import random

def execute_sql(sql, dbfullp, timeout=120):
    class QueryThread(threading.Thread):
        def __init__(self, sql, dbfullp):
            threading.Thread.__init__(self)
            self.sql = sql
            self.dbfullp = dbfullp
            self.result = None
            self.exception = None

        def run(self):
            conn, curs = self.connect_db(self.dbfullp)
            conn.text_factory = lambda b: b.decode(errors = 'ignore')
            try:
                conn.execute("BEGIN TRANSACTION;")
                curs.execute(self.sql)
                column_names = [description[0] for description in curs.description]
                self.result = pd.DataFrame(curs.fetchall(), columns=column_names)
            except Exception as e:
                self.exception = e
            curs.close()
            conn.close()
                
        def connect_db(self, dbfullp):
            db = dbfullp
            conn = sqlite3.connect(db)
            curs = conn.cursor()                
            return conn, curs

    query_thread = QueryThread(sql, dbfullp)
    query_thread.start()
    query_thread.join(timeout)
    if query_thread.is_alive():
        print(f"SQL query execution exceeded the timeout of {timeout} seconds.")
    if query_thread.exception:
        print(query_thread.exception)
    return query_thread.result

def df_normalization(df1, df2, num, type='ground'):
    cols1 = list(df1.columns)
    cols2 = list(df2.columns)

    target_col = set()
    for col1 in cols1:
        df1_type = str(df1[col1].dtype)
        if df1_type == 'object' and len(list(df1[col1])) - len(set(list(df1[col1]))) < 5:
            target_col = set(deepcopy(df1[col1]))
            break
    if len(target_col) == 0:
        for col1 in cols1:
            df1_type = str(df1[col1].dtype)
            if 'float' in df1_type:
                target_col = set(deepcopy(df1[col1]))
                break
    if len(target_col) == 0:
        for col1 in cols1:
            df1_type = str(df1[col1].dtype)
            if 'int' in df1_type:
                target_col = set(deepcopy(df1[col1]))
                break

    if len(target_col) == 0:
        return pd.DataFrame([])

    new_target_col = [] 
    for col2 in cols2:
        if df1[col1].dtype != df2[col2].dtype:
            continue
        if 'float' in str(df1[col1].dtype):
            count = 0
            for e in df2[col2]:
                for f in target_col:
                    if abs(e - f) < 0.005:
                        count += 1
                        new_target_col.append(e)
            if count >= len(target_col):
                new_target_col = list(set(new_target_col))
                break
        else:
            count = 0
            for e in df2[col2]:
                if e in target_col:
                    count += 1
            if count >= len(target_col):
                new_target_col = list(target_col)
                break

    if len(new_target_col) == 0:
        return pd.DataFrame([])

    if num == 1 and len(target_col) == len(set(target_col)):
        if type == 'predicted':
            dict1 = Counter(list(df1[col1]))
            dict2 = Counter(list(df2[col2]))
            check = True
            for a in dict1:
                if dict1[a] > dict2[a]:
                    check = False
                    break
            if check:
                return df1
        if new_target_col == list(target_col):
            return df2[df2[col2].isin(new_target_col)]
        else:
            return pd.DataFrame(df2[col2].unique(), columns=cols2) 
    else:
        return df2[df2[col2].isin(new_target_col)]

def check_dataframe(df1, df2):
    cols1 = df1.columns
    cols2 = df2.columns
    used_cols2 = set()
    count = 0
    for col1 in cols1:
        res = 0
        df1_type = str(df1[col1].dtype)
        if 'float' in df1_type:
            s1 = sorted(deepcopy(df1[col1].round(3)))
        elif df1_type == 'object':
            try:
                s1 = sorted(deepcopy(df1[col1].astype('int64')))
            except:
                s1 = sorted(deepcopy(df1[col1]))
        else:
            s1 = sorted(deepcopy(df1[col1]))

        for col2 in cols2:
            df2_type = str(df2[col2].dtype)
            if col2 in used_cols2:
                res += 1
                continue

            if 'float' in df2_type:
                s2 = sorted(deepcopy(df2[col2].round(3)))
            elif df2_type == 'object':
                try:
                    s2 = sorted(deepcopy(df2[col2].astype('int64')))
                except:
                    s2 = sorted(deepcopy(df2[col2]))
            else:
                s2 = sorted(deepcopy(df2[col2]))

            if 'float' in df1_type and 'float' in df2_type:
                s = True
                for i in range(len(s1)):
                    if abs(s1[i] - s2[i]) > 0.005:
                        s = False
                        break

                if s:
                    used_cols2.add(col2)
                    break
                else:
                    res += 1
            else:
                if s1 == s2:
                    used_cols2.add(col2)
                    break
                else:
                    res += 1

        if res < len(cols2):
            count += 1
    return (df1.shape[0], count)

   
def compare_onesql_outcomes(question, predsql, ground_truth, dbfullp, timeout=120, cutoff=30):
    print('==============================')
    print(dbfullp+' : ')
    print(question)
    try:
        temp_predicted_res = execute_sql(predsql, dbfullp, timeout=timeout)                
        ground_truth_res = ground_truth


        if 'full name' in question: # only for special questions asking about full name
            col_first = None
            for col in temp_predicted_res.columns:
                if 'first' in col:
                    col_first = col
                    break
            col_last = None
            for col in temp_predicted_res.columns:
                if 'last' in col:
                    col_first = col
                    break
            if col_first and col_last:
                temp_predicted_res['full_name'] = [temp_predicted_res.loc[i, col_first] + ' ' + temp_predicted_res.loc[i, col_last] for i in temp_predicted_res.index]

        cols = list(temp_predicted_res.columns)
        if len(cols) > len(set(cols)):
            new_cols = []
            for i in range(len(cols)):
                col = cols[i]
                if col not in new_cols:
                    new_cols.append(col)
                else:
                    new_cols.append(f"{col}_{i}")
            for i in range(len(cols)):
                temp_predicted_res.columns.values[i] = new_cols[i]
        for col in temp_predicted_res.columns:
            temp_predicted_res_type = str(temp_predicted_res[col].dtype)
            if 'float' in temp_predicted_res_type:
                temp_predicted_res[col] = temp_predicted_res[col].fillna(-10000000000.0)
            elif temp_predicted_res_type == 'object':
                try:
                    temp_predicted_res[col] = pd.to_numeric(temp_predicted_res[col], errors='raise').fillna(-10000000000.0)
                except:
                    try:
                        temp_predicted_res[col] = temp_predicted_res[col].fillna(str(-10000000000))
                        temp_predicted_res[col] = temp_predicted_res[col].astype('int64')
                    except:
                        temp_predicted_res[col] = temp_predicted_res[col].fillna('none')
            
                if 'true' in set(temp_predicted_res[col]) or 'false' in set(temp_predicted_res[col]):
                    try:
                        temp_predicted_res[col] = temp_predicted_res[col].str.lower().map({'true': True, 'false': False})
                    except:
                        pass
            
            elif 'int' in temp_predicted_res_type:
                temp_predicted_res[col] = temp_predicted_res[col].fillna(-10000000000)
                
        for col in temp_predicted_res.columns:
            temp_predicted_res_type = str(temp_predicted_res[col].dtype)
            if temp_predicted_res_type == 'object':
                temp_predicted_res.loc[temp_predicted_res[temp_predicted_res[col] == str(-10000000000)].index, col] = 'none'
        

        cols = list(ground_truth_res.columns)
        if len(cols) > len(set(cols)):
            new_cols = []
            for i in range(len(cols)):
                col = cols[i]
                if col not in new_cols:
                    new_cols.append(col)
                else:
                    new_cols.append(f"{col}_{i}")
            for i in range(len(cols)):
                ground_truth_res.columns.values[i] = new_cols[i]
        for col in ground_truth_res.columns:
            ground_truth_res_type = str(ground_truth_res[col].dtype)
            
            if 'float' in ground_truth_res_type:
                ground_truth_res[col] = ground_truth_res[col].fillna(-10000000000.0)
            elif ground_truth_res_type == 'object':
                try:
                    ground_truth_res[col] = pd.to_numeric(ground_truth_res[col], errors='raise').fillna(-10000000000.0)
                except:
                    try:
                        ground_truth_res[col] = ground_truth_res[col].fillna(str(-10000000000))
                        ground_truth_res[col] = ground_truth_res[col].astype('int64')
                    except:
                        ground_truth_res[col] = ground_truth_res[col].fillna('none')

                if 'true' in set(ground_truth_res[col]) or 'false' in set(ground_truth_res[col]):
                    try:
                        ground_truth_res[col] = ground_truth_res[col].str.lower().map({'true': True, 'false': False})
                    except:
                        pass
            
            elif 'int' in ground_truth_res_type:
                ground_truth_res[col] = ground_truth_res[col].fillna(-10000000000)
                
        for col in ground_truth_res.columns:
            ground_truth_res_type = str(ground_truth_res[col].dtype)
            if ground_truth_res_type == 'object':
                ground_truth_res.loc[ground_truth_res[ground_truth_res[col] == str(-10000000000)].index, col] = 'none'

        if temp_predicted_res.shape[0] < ground_truth_res.shape[0]:
            if ground_truth_res.shape[0] - temp_predicted_res.shape[0] > cutoff:
                res = (ground_truth_res.shape[0], 0)
            else:
                partial_ground_truth_res = pd.DataFrame([])
                partial_ground_truth_res = df_normalization(temp_predicted_res, ground_truth_res, ground_truth_res.shape[1], 'predicted')
                predicted_res = temp_predicted_res

                if predicted_res.shape[0] != partial_ground_truth_res.shape[0]:
                    res = (partial_ground_truth_res.shape[0], 0)
                else:
                    res = check_dataframe(partial_ground_truth_res, predicted_res)

        elif temp_predicted_res.shape[0] > ground_truth_res.shape[0]:
            if temp_predicted_res.shape[0] - ground_truth_res.shape[0] > cutoff:
                res = (ground_truth_res.shape[0], 0)
            else:
                predicted_res = pd.DataFrame([])
                predicted_res = df_normalization(ground_truth_res, temp_predicted_res, ground_truth_res.shape[1])
                if predicted_res.shape[0] > ground_truth_res.shape[0]:
                    predicted_res = predicted_res.drop_duplicates(keep='first')
                
                if predicted_res.shape[0] != ground_truth_res.shape[0]:
                    res = (ground_truth_res.shape[0], 0)
                else:
                    res = check_dataframe(ground_truth_res, predicted_res)

        elif temp_predicted_res.shape[0] == ground_truth_res.shape[0]:
            predicted_res = temp_predicted_res

            if predicted_res.shape[0] != ground_truth_res.shape[0]:
                res = (ground_truth_res.shape[0], 0)
            else:
                res = check_dataframe(ground_truth_res, predicted_res)

        print([temp_predicted_res.shape, ground_truth_res.shape, res])
        return [temp_predicted_res.shape, ground_truth_res.shape, res]
    
    except Exception as e:
        print(f"Error comparing SQL outcomes: {e}")

    return 0

def compare_results(question, predsql, ground_truth, dbfullp, timeout=120):
    try:
        res = func_timeout(timeout, compare_onesql_outcomes, args=(question, predsql, ground_truth, dbfullp))
        error = "incorrect answer" if res == 0 else "--"
    except FunctionTimedOut:
        print("Comparison timed out.")
        error = "timeout"
        res = [(-100, -100), (-100, -100), (-100, 0)]
    except Exception as e:
        print(f"Error in compare_results: {e}")
        error = str(e)
        res = [(-100, -100), (-100, -100), (-100, 0)]
    return {'exec_res': res, 'exec_err': error}


def major_voting(pred_sqls, dbfullp, return_random_one_when_all_errors=True):
    major_voting_counting = dict()
    # execute all sampled SQL queries to obtain their execution results
    execution_results=[execute_sql(sql, dbfullp, timeout=120) for sql in pred_sqls]

    # perform major voting
    # if all None, return random sql
    if all(x is None for x in execution_results):
        if return_random_one_when_all_errors:
            mj_pred_sql = random.choice(pred_sqls) # select a random one to return
        else:
            mj_pred_sql = "Error SQL"
    else: 
        for i, res in enumerate(execution_results):
            if res is not None: # skip invalid SQLs
                resstr=res.to_string()
                if resstr in major_voting_counting:
                    major_voting_counting[resstr]["votes"] += 1
                else:
                    major_voting_counting[resstr] = {"votes": 1, "sql": pred_sqls[i]}

        # find the SQL with the max votes
        major_vote = max(major_voting_counting.values(), key=lambda x: x["votes"])
        mj_pred_sql = major_vote["sql"]    
    return mj_pred_sql

def evaluate(mode, pred, dpath, outputpath, timeout=120):
    resdf=pd.read_json(pred).set_index('testid')
    resdf.drop(columns=['genout'], inplace=True)
    resdf['dbfullp']=resdf['dbname'].apply(lambda x: os.path.join(dpath, 'database', x, x+'.sqlite'))
    evalres={}
    for qid in tqdm(resdf.index.values):
        dbfullp = resdf.loc[qid,'dbfullp']
        question = resdf.loc[qid,'question']
        sqllst = resdf.loc[qid,'pred_sqls']

        evaluation_result = {
            "exec_res": [],
            "exec_err": [],
            "gold_result": [],
            "PREDICT_SQL": []
        }
        try:
            gcsvlst=ast.literal_eval(resdf.loc[qid,'goldcsv'])
        except:
            gcsvlst=[resdf.loc[qid,'goldcsv']]

        if mode=='greedy':
            predsql = sqllst[0]
        elif mode=='majority':
            predsql = major_voting(sqllst, dbfullp)
        else:
            print("Mode Error")
            predsql = None                

        if predsql is not None:
            for one_csv in gcsvlst:
                try:
                    gold_result = pd.read_csv(os.path.join(dpath, 'golden_csv', str(qid), one_csv))
                except:
                    gold_result = pd.DataFrame([])

                try:

                    response = compare_results(question, predsql, gold_result, dbfullp, timeout)

                    evaluation_result["exec_res"].append(response["exec_res"])
                    evaluation_result["exec_err"].append(response["exec_err"])

                except Exception as e:
                    evaluation_result["exec_res"].append("error")
                    evaluation_result["exec_err"].append(str(e))
                
                evaluation_result["gold_result"].append(gold_result.to_string(index=False))
                evaluation_result["PREDICT_SQL"].append(predsql)

        evalres[str(qid)] = evaluation_result
    return evalres

def select_prediction(res):
    pos = 0
    score = 0.0
    for i in range(len(res['exec_err'])):
        new_score = 0.0
        if res['exec_err'][i] == '--':
            if res['exec_res'][i][1] == res['exec_res'][i][2]:
                new_score = 1.0
            elif res['exec_res'][i][2][1] != 0:
                if res['exec_res'][i][1][0] == 0:
                    new_score = res['exec_res'][i][2][1] / res['exec_res'][i][1][1]
                else:
                    new_score = (res['exec_res'][i][2][0] * res['exec_res'][i][2][1]) / (res['exec_res'][i][1][0] * res['exec_res'][i][1][1])
        
        if new_score > score:
            score = new_score
            pos = i
    return pos


def analyzeeval(pred, outputpath, evalres):
    correct_qs, partial_correct_qs, wrong_qs = {}, {}, {}
    all_qs_points = {}
    for q in evalres:
        pos = 0
        if len(evalres[q]['exec_err']) > 1:
            pos = select_prediction(evalres[q])

        if evalres[q]['exec_err'][pos] == '--':
            if evalres[q]['exec_res'][pos] == None:
                all_qs_points[q] = 0.0
                wrong_qs[q] = ["completed wrong answer"]
            elif evalres[q]['exec_res'][pos][1] == evalres[q]['exec_res'][pos][2]:
                correct_qs[q] = [evalres[q]['PREDICT_SQL'][pos]]
                all_qs_points[q] = 1.0
            elif evalres[q]['exec_res'][pos][2][1] != 0:
                partial_correct_qs[q] = [evalres[q]['PREDICT_SQL'][pos]]
                if evalres[q]['exec_res'][pos][1][0] == 0:
                    all_qs_points[q] = evalres[q]['exec_res'][pos][2][1] / evalres[q]['exec_res'][pos][1][1]
                else:
                    all_qs_points[q] = (evalres[q]['exec_res'][pos][2][0] * evalres[q]['exec_res'][pos][2][1]) / (evalres[q]['exec_res'][pos][1][0] * evalres[q]['exec_res'][pos][1][1])
            else:
                all_qs_points[q] = 0.0
                wrong_qs[q] = ["completed wrong answer"]
        else:
            all_qs_points[q] = 0.0
            wrong_qs[q] = [evalres[q]['exec_err'][pos]]
            
    error_message = {}
    for q in wrong_qs:
        if wrong_qs[q][0] not in error_message:
            error_message[wrong_qs[q][0]] = [q]
        else:
            error_message[wrong_qs[q][0]].append(q)

    execacc=len(correct_qs) / len(evalres)
    partacc=sum(list(all_qs_points.values())) / len(evalres)
    
    print(f"correct SQL: {len(correct_qs)}, wrong SQL: {len(wrong_qs)}, partial correct SQL: {len(partial_correct_qs)}")
    print(f"accuracy: {execacc}, partial accuracy: {partacc}")
    for e in error_message:
        print(f"ERROR MESSAGE: '{e}', {len(error_message[e])}")

    return execacc, partacc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluations for Text2SQL.")
    parser.add_argument("--mode", type=str, choices=["greedy", "majority", "pass@k"])
    parser.add_argument("--pred", type=str, default="", help="Predicted result directory")
    parser.add_argument('--dpath', type = str, default = "")
    parser.add_argument('--db_type', type = str, default = "sqlite3")
    parser.add_argument('--outputpath', type = str, default = "./")

    args = parser.parse_args()
    evalres = evaluate(args.mode, args.pred, args.dpath, args.outputpath)
    analyzeeval(args.pred, args.outputpath, evalres)