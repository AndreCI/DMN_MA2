import argparse
import load_and_store as las
import task_1
import task_3

parser = argparse.ArgumentParser()
parser.add_argument('--babi_id', type=str, default="1", help='babi task ID')
parser.add_argument('--use_10k', type=bool, default=False, help='use 10k?')
args = parser.parse_args()



babi_train_raw, babi_test_raw = las.get_babi_raw_qa(args.babi_id, args.babi_id, args.use_10k)

print(type(babi_train_raw))



    
def generate_sentence(episode, id):
    sentence = []
    if(id=="1"):
        sentence.append(task_1.generate_pronoun(episode.get("Q"))) #pronoun
        sentence.append(task_1.generate_verb(episode.get("C")))
        sentence.append(task_1.generate_proposition())
        sentence.append(task_1.generate_determiner())
        sentence.append(task_1.generate_location(episode.get("A")))
    elif(id=="3"):
        sentence = task_3.generate_sentence(episode)
    else:
        raise Exception("Babi task not handled")
    return (' '.join(sentence)+'.')

babi_train_raw_new = []
babi_test_raw_new = []

while len(babi_train_raw)!=0:
    episode = babi_train_raw.pop()
    episode["A"] = generate_sentence(episode, args.babi_id)
    babi_train_raw_new.append(episode)
    
while len(babi_test_raw)!=0:
    episode = babi_test_raw.pop()
    episode["A"] = generate_sentence(episode, args.babi_id)
    babi_test_raw_new.append(episode)
    
las.init_write_babi(args.babi_id,babi_train_raw_new, "train", args.use_10k)
las.init_write_babi(args.babi_id, babi_test_raw_new, "test", args.use_10k)
