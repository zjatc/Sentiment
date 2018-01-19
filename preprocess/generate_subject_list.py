# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os


parser = argparse.ArgumentParser(description='extract subject list in a specific industry from excel file')
parser.add_argument('-if', '--input_file', default=None, help='File path of whole subject list')
parser.add_argument('-of', '--output_file', default=None, help='File path of output subject list')
parser.add_argument('-ind', '--industry', default=None, help='the target industry in which subject list will be extracted')
args = parser.parse_args()


subjects_dict = dict()


def add_in_subject_dict(subject_id, brand, word):
    subjects = set()
    if subject_id in subjects_dict.keys():
        subjects = subjects_dict[subject_id]
    subjects.add(brand.lower().replace('\t', ' '))
    subjects.add(word.lower().replace('\t', ' '))
    subjects_dict[subject_id] = subjects
    

def save_subject_list(output_file):
    folder_name = os.path.dirname(output_file)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print('Created folder with path ', folder_name)
    else:
        print('Folder already exists with path ', folder_name)
    
    lines = list()
    for sub_id in subjects_dict.keys():
        lines.append(sub_id + '\t'+'\t'.join(subjects_dict[sub_id]))
        
    open(output_file, 'w', encoding='utf-8', errors='ignore').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    subject_file = args.input_file
    output_file = args.output_file
    target_domain = args.industry
    
    subject_df = pd.read_excel(subject_file)
    print('loaded %d subjects' %subject_df.shape[0])
    domain_df = subject_df[subject_df.industry==target_domain]
    print('%d subjects in domain %s'%(domain_df.shape[0], target_domain))
    
    domain_df.apply(lambda r: add_in_subject_dict(r['subject_id'], r['brand'], r['word']), axis=1)
    
    save_subject_list(output_file)
    
