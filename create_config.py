import pandas as pd
import os
from tqdm import tqdm

peg_to_product = {}
prev_product = ''
prev = {}
for genome in os.listdir('OperonHunter-main/genomes/'):
    if genome[-3:] != 'gff':
        continue
    g = genome.split('.')[0]
    input = open('OperonHunter-main/genomes/' + genome, 'r')
    for row in input:
        if row[:3] != 'acc':
            continue
        cols = row.strip().split('\t')
        start = int(cols[3])
        end = int(cols[4])
        strand = cols[6]
        c = cols[8].split(';')
        if len(c) < 2 or 'locus' != c[1][:5]:
                continue
        product = c[1][10:]
        peg = c[0][3:]
        peg_to_product[product]  = peg

        prev[prev_product] = peg
        prev_product = peg
    input.close()

string = {} 
for string_file in os.listdir('OperonHunter-main/strings/'):
    if string_file[-3:] != 'txt':
        continue
    input = open('OperonHunter-main/strings/' + string_file, 'r')
    next(input)
    for row in input:
        cols = row.strip().split()
        g1 = cols[0].split('.')[1]
        g2 = cols[1].split('.')[1]
        if g1 in peg_to_product:
            g1 = peg_to_product[g1]
        else:
            continue
        if g2 in peg_to_product:
            g2 = peg_to_product[g2]
        else:
            continue
        if g1 in prev and g2 == prev[g1]:
            string[g1] = float(cols[2])/1000.0
    input.close()

data = pd.DataFrame(columns = ['name', 'im_set', 'path', 'score','split','label'])

for subdir, dirs, files in tqdm(os.walk('OperonHunter-main')):
    for file in files:
        if 'jpg' in file:
            name = file.split('_fig')[0]
            path = os.path.join('input_jsons', f'{name}.json')

            if 'test' in subdir.lower():
                split = 'test'
            if 'train' in subdir.lower():
                split = 'train'
            if 'valid' in subdir.lower():
                split = 'valid'

            if 'operon' in subdir.lower() and 'noperon' not in subdir.lower():
                label = 1
            elif 'noperon' in subdir.lower():
                label = 0

            im_set = subdir.split('images_')[-1].split('/')[0]

            score = 0.1
            if name in string:
                score = string[name]
            elif '85962.8' in name:
                temp = 'fig|85962.47.peg.' + name.split('.')[-1]
                if temp in string:
                    score = string[temp]

            data = pd.concat([data,pd.DataFrame({'name': name, 'im_set': im_set, 'path': path, 'score': score, 'split': split, 'label': label},index=[0])], ignore_index=True)

data.to_csv('data.csv', index=False)
