import sqlite3

connect = sqlite3.connect('user_database.db')

# allow python to execute sql command to a database
cur = connect.cursor() 

sql = """
CREATE TABLE faces (
    id integer primary key unique, 
    name text,
    balance integer
);
"""

# call the cursor to execute the command
cur.execute(sql) 

# agree to the database change or action
connect.commit()

# close database
connect.close()