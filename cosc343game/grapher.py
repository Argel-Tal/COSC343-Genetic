import matplotlib as plt
import numpy as np
import tensorflow as tf
import csv

stats = list()

with open('stats.csv', newline='\n') as csvfile:
    statFile = csv.reader(csvfile, delimiter=',')
    for row in statFile:
        stats.append(row)

print(stats[0])