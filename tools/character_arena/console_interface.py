import requests
import sqlite3
import json

DB_NAME = 'chat_history.db'

def create_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, character TEXT, user_input TEXT, response TEXT)''')
    conn.commit()
    conn.close()

def insert_into_table(character, user_input, response):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO history (character, user_input, response) VALUES (?, ?, ?)",
              (json.dumps(character), user_input, response))
    conn.commit()
    conn.close()

def generate_character(tcc_context, character_description):
    response = requests.post('http://localhost:5000/generate', json={
        'tcc_context': tcc_context,
        'character_description': character_description
    })
    return response.json()

def chat_with_character(character, tcc_context, scene_description, user_input):
    response = requests.post('http://localhost:5000/chat', json={
        'character': character,
        'tcc_context': tcc_context,
        'scene_description': scene_description,
        'human_input': user_input
    })
    return response.json()['response']

def main():
    create_table()
    tcc_context_file = input("Enter the path to the TCC context file: ")
    with open(tcc_context_file, 'r') as file:
        tcc_context = file.read()
    character_description = input("Enter character description: ")
    character = generate_character(tcc_context, character_description)
    print("Character generated.")
    scene_description = input("Enter scene description: ")
    print("Scene set")

    while True:
        user_input = input("Enter your message: ")
        response = chat_with_character(character, tcc_context, scene_description, user_input)
        print("Character response: ", response)
        insert_into_table(character, user_input, response)

if __name__ == '__main__':
    main()