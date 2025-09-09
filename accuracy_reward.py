def accuracy_reward(data_source, solution_str, ground_truth, extra_info=None):
    import pandas as pd
    from copy import deepcopy
    import threading
    import sqlite3
    from collections import Counter
    import re
    import pymysql
    
    sql_timeout = 120
    df_cutoff = 5
    pre_float = 0.005

    db = extra_info['dbfullp']
    dbname = extra_info['dbname']
    dbtype = extra_info['dbtype']
    csv = ground_truth

    host = extra_info['host']
    userid = extra_info['userid']
    pwd = extra_info['pwd']

    class QueryThread(threading.Thread):
        def __init__(self, sql, db, need_res=False):
            threading.Thread.__init__(self)
            self.sql = sql
            self.db = db
            self.result = None
            self.exception = None
            self.need_res = need_res

        def run(self):
            try:
                conn, curs = self.connect_db(self.db)
                conn.text_factory = lambda b: b.decode(errors = 'ignore')
                curs.execute(self.sql)
                
                if self.need_res:
                    column_names = [description[0] for description in curs.description]
                    self.result = pd.DataFrame(curs.fetchall(), columns=column_names)

                curs.close()
                conn.close()
            except Exception as e:
                self.exception = e
                
        def connect_db(self, db):
            if dbtype == 'MariaDB':
                conn = pymysql.connect(host=host, user= userid, password=pwd, db=dbname, charset='utf8')
                curs = conn.cursor(pymysql.cursors.DictCursor)
            elif dbtype == 'SQLite':
                conn = sqlite3.connect(db)
                curs = conn.cursor()
            return conn, curs
        
    # def df_normalization(df1, df2, num):
    def df_normalization(df1, df2, num, type='ground'):
        cols1 = list(df1.columns)
        cols2 = list(df2.columns)
        
        target_col = set()
        for col1 in cols1:
            df1_type = str(df1[col1].dtype)
            if df1_type == 'object':
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
                        if abs(e - f) < pre_float:
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

    def _compare_results_outcomes(temp_predicted_res, ground_truth_res):
        try:

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
                elif 'int' in temp_predicted_res_type:
                    temp_predicted_res[col] = temp_predicted_res[col].fillna(-10000000000)

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
                elif 'int' in ground_truth_res_type:
                    ground_truth_res[col] = ground_truth_res[col].fillna(-10000000000)

            if temp_predicted_res.shape[0] < ground_truth_res.shape[0]:
                if ground_truth_res.shape[0] - temp_predicted_res.shape[0] > df_cutoff:
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
                if temp_predicted_res.shape[0] - ground_truth_res.shape[0] > df_cutoff:
                    res = (ground_truth_res.shape[0], 0)
                
                else:
                    predicted_res = pd.DataFrame([])
                    predicted_res = df_normalization(ground_truth_res, temp_predicted_res, ground_truth_res.shape[1])
                        
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

            return [temp_predicted_res.shape, ground_truth_res.shape, res]
        
        except:
            return [temp_predicted_res.shape, ground_truth_res.shape, (0, 0)]
        
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
    
    # ==============================
    reward = 0
    try:
        sql = parse_response(solution_str)

        if len(sql) >= 16:
            run_sql = QueryThread(sql, db, need_res=True)
            run_sql.start()
            run_sql.join(sql_timeout)
            golden_df = pd.read_csv(csv)

            if not run_sql.is_alive() and run_sql.exception == None:
                golden_df = pd.read_csv(csv)
                res = _compare_results_outcomes(run_sql.result, golden_df)
                if res[1][0] != 0:
                    acc = (res[2][0] * res[2][1]) / (res[1][0] * res[1][1])
                else:
                    acc = res[2][1] / res[1][1]
                
                if acc == 0:
                    reward = 0.5
                else:
                    reward = 10 * acc
            else:
                reward = 0
        else:
            reward = 0
    except:
        reward = 0

    return reward
