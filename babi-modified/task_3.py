


"""
exemple:
39 Where was the apple before the bathroom? 	office	38 25 22
goal:
The apple was in the office before being in the bathroom
"""
def generate_subject(question):
    question = question.split(" ")
    return question[3]

def generate_place(question):
    question = question.split(" ")
    return question[len(question)-1]

def generate_sentence(episode):
    sentence = []
    sentence.append("The")
    
    sentence.append(generate_subject(episode.get("Q")))    
    
    sentence.append("was")
    sentence.append("in")
    sentence.append("the")
    
    sentence.append(episode.get("A"))
    
    sentence.append("before")
    sentence.append("being")
    sentence.append("in")
    sentence.append("the")
    sentence.append(generate_place(episode.get("Q")))
    
    return sentence