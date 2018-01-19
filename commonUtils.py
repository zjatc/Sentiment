# -*- coding:utf-8 -*-

import sys
import time
import os
import shutil
from datetime import timedelta

''' get time difference from now to the given start timestamp '''
def get_spent_times(start_time):
    end_time = time.time()
    spent_time = end_time - start_time
    return timedelta(seconds=int(round(spent_time)))


''' common file reader, mode: 'r' for read and 'w' for write '''
def open_file(fileName, mode='r'):
    return open(fileName, mode, encoding='utf-8', errors='ignore')


def check_file_exist(filePath, fileType='', need_return=False):
    if not os.path.exists(filePath):
        if need_return:
            return False
        else:
            raise FileNotFoundError("""Given {0} file doesn't existed! {1}""" .format(fileType, filePath))
    return True

def recreate_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    print('Recreated folder with path ', path)

def create_folder(filePath):
    folder_name = os.path.dirname(filePath)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print('Created folder with path ', folder_name)
    else:
        print('Folder already exists with path ', folder_name)
