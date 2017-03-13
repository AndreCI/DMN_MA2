import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from utils.data_utils import load_glove, WordTable
import read_data as rd

data_path = 'data/tasks_1-20_v1-2/en-10k'
wo = load_glove(50)
words = WordTable(wo,50)
train = rd.read_babi(data_path,2,'train',128,words)

