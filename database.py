import sqlite3

def connect_database():
    conn = sqlite3.connect('object_detection_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS object_detection_data (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_input TEXT,
                      object_name TEXT
                   )''')
    conn.commit()
    return conn, cursor

def insert_data(conn, cursor, user_input, object_name):
    cursor.execute("INSERT INTO object_detection_data (user_input, object_name) VALUES (?, ?)", (user_input, object_name))
    conn.commit()

def retrieve_data(conn, cursor):
    cursor.execute("SELECT user_input, object_name FROM object_detection_data")
    data = cursor.fetchall()
    conn.close()
    return data

def retrieve_user_input(conn, cursor):
    cursor.execute("SELECT user_input FROM object_detection_data ORDER BY id DESC LIMIT 1")
    user_input_data = cursor.fetchone()
    conn.close()
    if user_input_data:
        return user_input_data[0]
    else:
        return None

