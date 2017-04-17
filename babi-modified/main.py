import argparse
import load_and_store as las
from random import randint

parser = argparse.ArgumentParser()
parser.add_argument('--babi_id', type=str, default="1", help='babi task ID')
parser.add_argument('--babi_test_id', type=str, default="", help='babi_id of test set (leave empty to use --babi_id)')
args = parser.parse_args()



babi_train_raw, babi_test_raw = las.get_babi_raw_qa(args.babi_id, args.babi_test_id)

#babi_name = "1"
#jack = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output_data/en/%s_train.txt' % babi_name), 'a')

def generate_pronoun(question):
    return question[9:len(question)]

def generate_verb(facts):
    facts = facts[0:len(facts)-2]
    sentences = facts.split('. ')
    verbs =[]
    for s in sentences:
        words = s.split(' ')
        if words[1] not in verbs:
            verbs.append(words[1])
    random = randint(0,len(verbs)-1)
    return verbs[random]
    
def generate_proposition():
    return "to"

def generate_determiner():
    return "the"
    
def generate_location(answer):
    return answer
    
def generate_sentence(episode):
    sentence = []
    sentence.append(generate_pronoun(episode.get("Q"))) #pronoun
    sentence.append(generate_verb(episode.get("C")))
    sentence.append(generate_proposition())
    sentence.append(generate_determiner())
    sentence.append(generate_location(episode.get("A")))
    
    return (' '.join(sentence)+'.')


def generate_sentence_simple(episode):
    sentence = {'pronoun':'','verb':'','preposition':'','determiner':'','location':''}
    q = episode.get("Q")
    sentence['pronoun'] = q[9:len(q)]
    sentence['verb'] = "is"
    sentence['proposition'] = "in"
    sentence['determiner'] = "the"
    answer = episode.get("A")
    sentence['location'] = answer
    return sentence

def reconstruct_sentence_simple(dic):
    sentence=[]
    sentence.append(dic['pronoun'])
    sentence.append(dic['verb'])
    sentence.append(dic['proposition'])
    sentence.append(dic['determiner'])
    sentence.append(dic['location'])
    return (' '.join(sentence)+'.')

babi_train_raw_new = []
babi_test_raw_new = []
while len(babi_train_raw)!=0:
    episode = babi_train_raw.pop()
    episode["A"] = generate_sentence(episode)
    babi_train_raw_new.append(episode)
    episode = babi_test_raw.pop()
    episode["A"] = generate_sentence(episode)
    babi_test_raw_new.append(episode)
las.init_write_babi("1",babi_train_raw_new, "train")
las.init_write_babi("1", babi_test_raw_new, "test")
