# -*- coding:utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import numpy as np
from hanziconv import HanziConv


arg_parser = argparse.ArgumentParser(description='Dealing with P2P_news datset for subject-based sentiment analysis experiments.')
arg_parser.add_argument('-i', '--input', default=None, help='path of the input data file')
arg_parser.add_argument('-o', '--output', default=None, help='path of the output data file')
arg_parser.add_argument('-s', '--subject', default=None, help='path of the subjects list file')
args = arg_parser.parse_args()


class SubjectHelper(object):
    '''
    dealing with subject dictionary
    '''
    def __init__(self, subjectFile):
        self.subjctFile = subjectFile
        

    def load_subjects(self):
        self.subject_list = []
        with open(self.subjctFile, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                tmp = line.strip().split('\t')
                sId = tmp[0]
                if sId=='q0':
                    continue
                subjs = list(set([HanziConv.toSimplified(x).strip().lower() for x in tmp[1:]]))
                subjs.sort(key=lambda x: len(x), reverse=True)
                self.subject_list.append(subjs)
        print('loaded %d subjects' % len(self.subject_list))


    def get_subject_index(self, text, target):
    
        def get_potential_subject(target):
            for subjs in self.subject_list:
                if target in subjs:
                    return subjs
            return list()
    
        subjs = get_potential_subject(target.strip().lower())
        for subj in subjs:
            s_idx = text.lower().find(subj)
            if s_idx == -1:
                continue
            e_idx = s_idx + len(subj)
            return '_'.join([str(s_idx), str(e_idx)])
    
        return '-1_-1'


def load_data(inFile):
    df = pd.read_csv(inFile, quotechar='"', header=0)
    print('loaded %d instances with %d columns.' % (df.shape[0], df.shape[1]))
    return df


def generate_target_index(df, subjHelper):
    df['indices'] = df.apply(lambda x: subjHelper.get_subject_index(str(x['text']), str(x['target'])), axis=1)
    return df


def write_to_file(df, outputFile):
    text = df.pop('text')
    indices = df.pop('indices')
    label = df.pop('label')
    df.insert(0, 'text', text)
    df.insert(1, 'indices', indices)
    df.insert(2, 'label', label)
    
    cols = df.columns
    print('total %d instances with columns: "%s", "%s", "%s", "%s"' % (df.shape[0], cols[0], cols[1], cols[2], cols[3]))
    labelNum = df.groupby('label').size()
    print('includes %d positive, %d negative, %d neutral instances' % (labelNum[1], labelNum[-1], labelNum[0]))
    
    df.to_csv(outputFile, index=False, header=False, quotechar='"')


if __name__ == '__main__':

	assert args.input is not None
	assert args.output is not None
	assert args.subject is not None

	input_file = args.input
	output_file = args.output
	subject_file = args.subject

	subjHelper = SubjectHelper(subject_file)
	subjHelper.load_subjects()

	df = load_data(input_file)
	df = generate_target_index(df, subjHelper)

	df_final = df[df.indices!='-1_-1']
	write_to_file(df_final, output_file)
