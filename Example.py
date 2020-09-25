import pandas as pd
from MarkovModel import MarkovModel

import csv
import random

# making a fake loaded die training set

def make_e_and_h():
    n = random.randint(8, 12)
    e = ''
    h = 's'
    for i in range(n):
        if bool(random.getrandbits(1)):
            e += 'H' if random.random() < 0.5 else 'T'
            h += 't'
        else:
            e += 'H' if random.random() < 0.7 else 'T'
            h += 'f'
    h += 'e'
    return[e,h]

training_file = "C:/Users/ggoh/Documents/train.csv"

with open(training_file, 'w', newline='') as output:
    writer = csv.writer(output)
    writer.writerow(['emission', 'hidden'])
    for i in range(500):
        writer.writerow(make_e_and_h)


# making a viterbi test and training it

viterbi_test = "C:/Users/ggoh/Documents/predict_by_viterbi_test.csv"

with open(viterbi_test, 'w', newline='') as output:
    writer = csv.writer(output)
    writer.writerow(['emission', 'hidden', 'predicted'])
    for i in range(50):
        writer.writerow(make_e_and_h)

training_df = pd.read_csv(viterbi_test)
HMM = MarkovModel(training_df)

# testing

testing_df = pd.read_csv(viterbi_test)

for index,row in testing_df.iterrows():
    testing_df.loc[index, 'predicted'] = HMM.predict(row['emission'])

testing_df.to_csv(viterbi_test)

# making a rules test and training it

rules_test = "C:/Users/ggoh/Documents/predict_by_rules_test.csv"

with open(rules_test, 'w', newline='') as output:
    writer = csv.writer(output)
    writer.writerow(['emission', 'hidden', 'predicted'])
    for i in range(50):
        writer.writerow(make_e_and_h)

training_df = pd.read_csv(rules_test)
HMM = MarkovModel(training_df)

# testing

testing_df = pd.read_csv(rules_test)

def rules(sequence):
    return abs(sequence.count('t') - sequence.count('f')) <= 1

for index,row in testing_df.iterrows():
    testing_df.loc[index, 'predicted'] = HMM.predict(row['emission'], method = 'predict_by_rules', constraints=[rules])

print(testing_df)

testing_df.to_csv(rules_test)