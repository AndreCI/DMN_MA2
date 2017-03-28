

def count_qa(text):
    dicQ={}
    dicA={}
    while len(text)!=0:
        episode = text.pop()
        q = episode.get("Q")
        if dicQ.has_key(q):
            dicQ[q] = dicQ[q]+1
        else:
            dicQ[q] = 1
        a = episode.get("A")
        if(dicA.has_key(a)):
            dicA[a] = dicA[a]+1
        else:
            dicA[a] = 1
    return dicQ, dicA