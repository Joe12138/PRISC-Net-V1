import csv
from typing import List

def init_csv_stat(save_loc: str, items: List[str]):
    with open(save_loc, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(items)

def save_csv_stat(save_loc: str, stat: List):
    with open(save_loc, 'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(stat)