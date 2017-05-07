from random import randint


def generate_pronoun(question):
    return question[9:len(question)]

def generate_verb_random(facts):
    facts = facts[0:len(facts)-2]
    sentences = facts.split('. ')
    verbs =[]
    for s in sentences:
        words = s.split(' ')
        if words[1] not in verbs:
            verbs.append(words[1])
    random = randint(0,len(verbs)-1)
    return verbs[random]

def generate_verb(facts):
    return "went"

def generate_proposition():
    return "to"

def generate_determiner():
    return "the"
    
def generate_location(answer):
    return answer
    
    
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