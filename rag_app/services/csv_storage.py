# csv_storage.py
import csv
import os
from typing import Optional

CSV_FILE_PATH = "answered_questions.csv"

def get_stored_answer(question_id: str) -> Optional[str]:
    """
    Check if a question with the given ID has already been answered.
    Returns the stored answer if found, otherwise None.
    """
    if not os.path.exists(CSV_FILE_PATH):
        return None
    with open(CSV_FILE_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("id") == question_id:
                return row.get("answer")
    return None

def store_answer(question_id: str, question: str, answer: str) -> None:
    """
    Append a new record with the question ID, question text, and answer into the CSV file.
    If the file does not exist, it creates it with a header.
    """
    file_exists = os.path.exists(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["id", "question", "answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"id": question_id, "question": question, "answer": answer})