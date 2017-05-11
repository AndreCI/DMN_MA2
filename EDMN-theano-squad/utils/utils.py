import os as os
import numpy as np
import json
import math


babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "all_shuffled", 
        "sh1": "../shuffled/qa1_single-supporting-fact",
        "sh2": "../shuffled/qa2_two-supporting-facts",
        "sh3": "../shuffled/qa3_three-supporting-facts",
        "sh4": "../shuffled/qa4_two-arg-relations",
        "sh5": "../shuffled/qa5_three-arg-relations",
        "sh6": "../shuffled/qa6_yes-no-questions",
        "sh7": "../shuffled/qa7_counting",
        "sh8": "../shuffled/qa8_lists-sets",
        "sh9": "../shuffled/qa9_simple-negation",
        "sh10": "../shuffled/qa10_indefinite-knowledge",
        "sh11": "../shuffled/qa11_basic-coreference",
        "sh12": "../shuffled/qa12_conjunction",
        "sh13": "../shuffled/qa13_compound-coreference",
        "sh14": "../shuffled/qa14_time-reasoning",
        "sh15": "../shuffled/qa15_basic-deduction",
        "sh16": "../shuffled/qa16_basic-induction",
        "sh17": "../shuffled/qa17_positional-reasoning",
        "sh18": "../shuffled/qa18_size-reasoning",
        "sh19": "../shuffled/qa19_path-finding",
        "sh20": "../shuffled/qa20_agents-motivations",
    }


def init_squad(fname, test_pourcentage = 0.2):
    '''
    Load data from fname
    '''
    print("==> Loading data from %s (SQUAD)" %fname)
    tasks_train = []
    tasks_test = []
    task = None
    with open(fname) as data_json:
        data = json.load(data_json)
        data = data['data']
        for i in range(0, np.shape(data)[0]):
            paragraph = data[i]['paragraphs']
            size = np.shape(paragraph)[0]
            test_br = math.floor(size*test_pourcentage)         
            
            for j in range(0, np.shape(paragraph)[0]):
                current = paragraph[j]
                context = current['context']
                context = context.replace('.', ' . ')
                qas = current['qas']
                for k in range(0, np.shape(qas)[0]):
                    task = {"C": "", "Q": "", "A":""}
                    
                    current_qas = qas[k]
                    question = current_qas['question']
                    answer = current_qas['answers'][0]['text']
                    answer = answer.replace('?', '')
                    task["A"] = answer.encode("utf-8")
                    task["C"] = context.encode("utf-8")
                    task["Q"] = question.encode("utf-8")
                    if(j<test_br):
                        tasks_test.append(task.copy())
                    else:
                        tasks_train.append(task.copy())
                
    return tasks_train, tasks_test

def get_squad_raw():
    return init_squad(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/squad/train-v1.1.json'))


def init_babi(fname):
    '''
    Load data from fname
    :param fname: the path where the data are
    :return tasks: raw data from fname.
    '''
    print("==> Loading data from %s" % fname)
    tasks = []
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": ""} 
            
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        if line.find('?') == -1:
            task["C"] += line
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            tasks.append(task.copy())

    return tasks


def get_babi_raw(id, test_id, nbr_k, multiple=False):
    '''
    Basic getter function to load the data set.
    :param id: the number the task to train or test
    :param test_id: Not sure why it is useful. If test_id="", takes the value id
    :return babi_train_raw, babi_test_raw: the data loaded
    '''

    if (test_id == ""):
        test_id = id 
    babi_name = babi_map[id]
    babi_test_name = babi_map[test_id]
    if(multiple):
        if(nbr_k==100):
            babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-100k-m/%s_train.txt' % babi_name))
            babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-100k-m/%s_test.txt' % babi_test_name))
        elif(nbr_k==10):
            babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k-m/%s_train.txt' % babi_name))
            babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k-m/%s_test.txt' % babi_test_name))
        else:
            babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-m/%s_train.txt' % babi_name))
            babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-m/%s_test.txt' % babi_test_name))
    else:
        if(nbr_k==100):
            babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-100k/%s_train.txt' % babi_name))
            babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-100k/%s_test.txt' % babi_test_name))
        elif(nbr_k==10):
            babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k/%s_train.txt' % babi_name))
            babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k/%s_test.txt' % babi_test_name))
        else:
            babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en/%s_train.txt' % babi_name))
            babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en/%s_test.txt' % babi_test_name))
    return babi_train_raw, babi_test_raw

            
def load_glove(dim):
    '''
    Load GloVe from google.
    :param dim: Word embedding dimension. Must be 50, 100, 150, 200.
    :return word2vec: a dictionary containing the word embedding. 
    '''
    #TODO: Check it this is right?
    word2vec = {}
    
    print "==> loading glove"
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
            
    print "==> glove is loaded"
    
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=False):
    '''
    Create a word embedding for given word. Really not perfect.
    if the word is missing from Glove, create some fake vector and store in glove!
    :param word: a word in raw text
    :param word2vec: the word embedding matrix
    :param word_vector_size: size of word embedding (50,100,200,300)
    :param silent: if False, this function will print a warning to tell the user that the word is not in word2vec.
    :return vector: the vector created for the given word
    '''
    
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print("utils/utils.py::create_vector => Warning: %s is missing" % word)
    return vector


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
    '''
    Check if the word is in word2vec & vocab, and if not, add it.
    Return the word representation depending on to_return option
    :param word: a word, in raw text
    :param word2vec: embedding matrix containing all the words known
    :param vocab: list of all words known. Given a word, return the index
    :param ivocab: inverse of vocab, i.e. list of all word, given an index, return a word
    :param word_vector_size: size of word embedding (50, 100, 200, 300)
    :param to_return: option to choose what the function should return.
    :param silent: used in case word is not in word2vec. If False, will print a warning
    :return: word2vec[word] or vocab[word]
    '''
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet.")
    else:
        raise Exception("to_return =",to_return,"is unknow. Please change to word2vec or index.")

def get_word(word2vec, emb):
    for (k,v) in word2vec.iteritems():
        if((v==emb).all()):
            return k
    return "<wnf>"

def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)