import argparse
import load_and_store as las


parser = argparse.ArgumentParser()
parser.add_argument('--babi_id', type=str, default="1", help='babi task ID')
parser.add_argument('--babi_test_id', type=str, default="", help='babi_id of test set (leave empty to use --babi_id)')
args = parser.parse_args()



babi_train_raw, _ = las.get_babi_raw_qa(args.babi_id, args.babi_test_id)


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

while len(babi_train_raw)!=0:
    a = babi_train_raw.pop()
    dic = generate_sentence(a)
    print(reconstruct_sentence(dic))
    