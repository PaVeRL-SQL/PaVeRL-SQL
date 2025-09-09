import os
import json
import pandas as pd

class Read_Data:
    def run(self, args):
        df = pd.read_csv(args['data_path'])
        
        all_prompts, all_qs, all_dbs = {}, {}, {}
        for i in df.index:
            all_qs[df.loc[i, 'testid']] = df.loc[i, 'question']
            all_prompts[df.loc[i, 'testid']] = df.loc[i, 'promptin']
            all_dbs[df.loc[i, 'testid']] = os.path.join( args['db_path'], f"{df.loc[i, 'dbname']}/{df.loc[i, 'dbname']}.sqlite")
        
        return all_qs, all_prompts, all_dbs
