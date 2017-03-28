import argparse
import load_and_store as las
import os

parser = argparse.ArgumentParser()
parser.add_argument('--babi_id', type=str, default="1", help='babi task ID')
parser.add_argument('--babi_test_id', type=str, default="", help='babi_id of test set (leave empty to use --babi_id)')
args = parser.parse_args()



babi_train_raw, _ = las.get_babi_raw_qa(args.babi_id, args.babi_test_id)

#babi_name = "1"
#jack = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output_data/en/%s_train.txt' % babi_name), 'a')


def generate_sentence(episode):
    sentence = {'pronoun':'','verb':'','preposition':'','determiner':'','location':''}
    q = episode.get("Q")
    sentence['pronoun'] = q[9:len(q)]
    sentence['verb'] = "is"
    sentence['proposition'] = "in"
    sentence['determiner'] = "the"
    answer = episode.get("A")
    sentence['location'] = answer
    return sentence

def reconstruct_sentence(dic):
    sentence=[]
    sentence.append(dic['pronoun'])
    sentence.append(dic['verb'])
    sentence.append(dic['proposition'])
    sentence.append(dic['determiner'])
    sentence.append(dic['location'])
    return (' '.join(sentence)+'.')

babi_train_raw_new = []
while len(babi_train_raw)!=0:
    episode = babi_train_raw.pop()
    dic = generate_sentence(episode)
    episode["A"] = reconstruct_sentence(dic)
    babi_train_raw_new.append(episode)
las.init_write_babi("1",babi_train_raw_new)
