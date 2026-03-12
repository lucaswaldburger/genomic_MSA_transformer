# 01 Sanitize ecoli and bsub shell output

import re
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def filter_data(x):
    noise_patterns = [
        'Epoch', 'lwaldburger', 'Better model found', '█', 'Script',
        '[?2004', 'exit', '^[[B^[[B^C0+1', '0+0', '0 bytes copied',
    ]
    if any(p in x for p in noise_patterns) or x in ('c', 'd'):
        return False
    return True


def sanitize_row(x):
    str_builder = x
    for re_match in re.findall(r'[-]+', x):
        str_builder = str_builder.replace(re_match, ' ')
    return str_builder


def add_model(df):
    models = [
        '+ horizontal flip + zoom',
        '- horizontal flip + zoom',
        '+ horizontal flip - zoom',
        '- horizontal flip - zoom',
    ]
    curr_model = 0

    df['model'] = None
    for idx, row in df.iterrows():
        df.loc[idx, 'model'] = models[curr_model]
        if row['epoch'] == 'epoch':
            curr_model += 1
    df = df[df['epoch'] != 'epoch'].reset_index(drop=True)
    return df


def parse_log(log_path):
    df = pd.read_csv(log_path, sep='\t', names=['text'])
    df = df[df['text'].apply(lambda x: filter_data(x))].reset_index(drop=True)
    df['text'] = df['text'].apply(lambda x: x.replace(' ', '-'))
    df['text'] = df['text'].apply(lambda x: sanitize_row(x))
    df.to_csv('temp.csv', index=False, header=False)
    df = pd.read_csv('temp.csv', sep=' ', index_col=False).drop(
        ['Unnamed: 5', 'Unnamed: 6'], axis=1)
    df = add_model(df)
    return df


ecoli_df = parse_log('outputs/ecoli/ecoli_history_log.txt')
ecoli_df['organism'] = 'E. coli'
bsub_df = parse_log('outputs/bsub/bsubtilis_history_log.txt')
bsub_df['organism'] = 'B. subtilis'
df = pd.concat([ecoli_df, bsub_df])


# 02 Plot model performance

df['accuracy'] = df['accuracy'].astype(float)
df['epoch'] = df['epoch'].astype(float)
df['train_loss'] = df['train_loss'].astype(float)
df['valid_loss'] = df['valid_loss'].astype(float)

g = sns.relplot(kind='line', data=df, hue='model', x='epoch', col='organism', y='accuracy')
g.set(xlabel='Epoch', ylabel='Accuracy')

g = sns.relplot(kind='line', data=df, hue='model', x='epoch', col='organism', y='train_loss')
g.set(xlabel='Epoch', ylabel='Training Loss')

g = sns.relplot(kind='line', data=df, hue='model', x='epoch', col='organism', y='valid_loss')
g.set(xlabel='Epoch', ylabel='Validation Loss')

df['organism_model'] = df['organism'] + ' | ' + df['model']

plt.figure(figsize=(12, 4))
df['Organism'] = df['organism']
g = sns.swarmplot(data=df, x='model', y='accuracy', hue='Organism', dodge=True)
g.set(ylabel='Accuracy')
fig = g.get_figure()
fig.savefig('ablation.svg', dpi=300)


# 03 Classifier performance

TOTAL_ECOLI_OPERONS = None
TOTAL_ECOLI_NOPERONS = None
TOTAL_BSUB_OPERONS = None
TOTAL_BSUB_NOPERONS = None

file_dict = {}
for txt_file in glob('outputs/ecoli/*pegs*'):
    filename = txt_file.split('/')[-1]
    with open(txt_file) as f:
        peg_list = [line.strip() for line in f.readlines()]
    file_dict[filename] = peg_list

pd.DataFrame.from_dict(file_dict, orient='index')

# TODO
# import dataframe where pegs are a list
# make venn diagrams of ablation
# calculate operon prediction pegs for ecoli and bsub
