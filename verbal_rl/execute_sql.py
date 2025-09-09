import threading
import pandas as pd
import sqlite3

class QueryThread(threading.Thread):
    def __init__(self, sql, db):
        threading.Thread.__init__(self)
        self.sql = sql
        self.db = db
        self.result = None
        self.exception = None

    def run(self):
        try:
            conn, curs = self.connect_db(self.db)
            curs.execute(self.sql)
            self.result = pd.DataFrame(curs.fetchall())

            curs.close()
            conn.close()
        except Exception as e:
            self.exception = e
            
    def connect_db(self, db):
        conn = sqlite3.connect(db)
        curs = conn.cursor()
        return conn, curs