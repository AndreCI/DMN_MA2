# DMN_MA2

Question Answering is an important and growing field in the Natural Language Processing domain, which concern multiple and diverse real life applications.
Dynamic Memory Networks are a new and powerful way to handle the Question Answering problem. They are a neural network architecture which learns to answer questions by using raw input-question-answer triplets. However, they only produce one-word answer and thus cannot be easily adapted to construct chat-bots. In this paper and repo, two major modifications are made to allow the model to produces multiple-words answer. The first modification, inspired by the Encoder-Decoder model, gives great results but fails to scale up with difficult tasks. The second modification, the Pointer Net architecture, is an interesting direction to look at but fails to solve this problem on its own.

Repo for my master project, 12 credit HTC.

## Repository contents

| file | description |
| --- | --- |
| `EDMN-theano` | The code itself |
| `bAbI-tasks-master` | code from facebook to produce the data sets |
| `babi-modified` | The code created to modifiy the bAbI tasks |
| `report` | Contains the different presentations, semestrer report, report ressources |
| `sandbox.ipynb` | Jupyter sandbox used to generate the figures |

Director: Mi Fei
Supervisor: Boi Faltings
Student: André Cibils

The code comes from YerevaNN (https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano). I reworded it a bit to make it work, then I modified it quite a lot to solve the problem posed by this project.

