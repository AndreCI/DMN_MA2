import os as os
import numpy as np

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

def write_babi(fname,data):
    file_obj = open(fname, "w")
    line_count = 1
    for i in range(0,np.shape(data)[0]):             
        episode = data.pop()
        facts = episode["C"].split(".")
        question = episode["Q"]
        answer = episode["A"]
        j=0
        for j in range(0, np.shape(facts)[0]-1):
            line = ""
            if j==0:
                line = str(line_count) + " "
            else:
                line = str(line_count)
            line = line + facts[j] + "."
            line = line.replace(' .', '.')
            line = line + "\n"
            file_obj.write(line)
            line_count = line_count+1
        line = ""
        line = str(line_count) + " "+question + "? \t" + answer
        line = line + "\n"
        file_obj.write(line)
        line_count = line_count + 1
        if line_count>15:
            line_count=1
    print("writing finished.")
    
    
def init_write_babi(id,data, mode, k_nbr):
    babi_name = babi_map[id]
    if(k_nbr==100):
        if(mode == "train"):
            write_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output_data/en-100k/%s_train.txt' % babi_name), data)
        else:
            write_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output_data/en-100k/%s_test.txt' % babi_name), data)
    elif(k_nbr==10):
        if(mode == "train"):
            write_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output_data/en-10k/%s_train.txt' % babi_name), data)
        else:
            write_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output_data/en-10k/%s_test.txt' % babi_name), data)
    else:
        if(mode == "train"):
            write_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output_data/en/%s_train.txt' % babi_name), data)
        else:
            write_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output_data/en/%s_test.txt' % babi_name), data)
        

def init_babi(fname):
    tasks = []
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C":"","Q": "", "A": ""} 
            
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


def get_babi_raw_qa(id, test_id, nbr_k):
    if (test_id == ""):
        test_id = id 
    babi_name = babi_map[id]
    babi_test_name = babi_map[test_id]
    if(nbr_k==100):
        babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'original_data/en-100k/%s_train.txt' % babi_name))
        babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'original_data/en-100k/%s_test.txt' % babi_test_name))
    elif(nbr_k==10):
        babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'original_data/en-10k/%s_train.txt' % babi_name))
        babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'original_data/en-10k/%s_test.txt' % babi_test_name))
    else:
        babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'original_data/en/%s_train.txt' % babi_name))
        babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'original_data/en/%s_test.txt' % babi_test_name))

    return babi_train_raw, babi_test_raw
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    