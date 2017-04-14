import random
import numpy as np

import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
from lasagne import layers
from lasagne import nonlinearities
import cPickle as pickle




#alpha = T.ivector()
#beta = T.ivector()
#delta = T.ivector()
#losange = T.ivector()
#
#def square(a):
#    return a*a
#    
#outputs, updates = theano.scan(fn = square, sequences = alpha, n_steps = 5)
#beta = outputs
#
#losange = beta + delta
#
#calcul = theano.function(inputs = [alpha, delta], outputs = [beta, losange], allow_input_downcast = True)
#
#ada = np.arange(5)
#delt = np.arange(5) +2
#
#jack = calcul(ada, delt)
#print(jack)
#
#
#
#print("-------------------------")
#exit()
#
#print("Lauchning test")
#
#ans = T.ivector()
#pred = T.ivector()
#
#def calc(curr_ans, curr_pred, loss):
#    print(curr_ans)
#    temp= (loss + curr_ans*curr_pred)
#    print(temp)
#    return temp
#
#outputs,updates = theano.scan(fn = calc, 
#                              sequences = [ans, pred], outputs_info = [np.int32(0)], n_steps = 5)
#
#loss_ce = outputs
#
#loss_cal = theano.function(inputs = [ans, pred], outputs = [loss_ce], allow_input_downcast = True)
#
#a = np.arange(5)
#b = np.arange(5)
#
#c = loss_cal(a,b)
#print(c)
#exit()


print("--------------------")
answers = T.ivector()
temp = T.scalar()
predictions = T.matrix()

def loss_temp(curr_ans,curr_pred, loss):
    T.printing.Print(curr_ans)
    T.printing.Print(curr_pred)
    T.printing.Print(loss)
    temp= T.nnet.categorical_crossentropy(curr_pred.dimshuffle('x',0), T.stack([curr_ans]))[0]
    return [temp + loss]



outputs, updates = theano.scan(fn = loss_temp, 
                               sequences = [answers, predictions], 
                               outputs_info = [np.float64(0.0)], 
                                               n_steps = 5)

loss_ce = outputs

loss_cal = theano.function(inputs = [answers, predictions], outputs = [loss_ce])
#loss2 = theano.function(inputs = [answers, predictions], outputs = [loss_test])
#END DECLARATION
#START VAR DEC AND COMPUTATION
max_nbr = 5
pred = []
for i in range(0, max_nbr):
    temp = np.ones(8)
    temp[i] = temp[i] + 5
    temp = temp/sum(temp)
    pred.append(temp)

#print(pred)

answers = []
for i in range(0, max_nbr):
    answers.append(pred[i].argmax())
#print(answers)

loss = loss_cal(answers, predictions)
print(loss)

exit()


predictions = T.vector()
answer = T.iscalar()

predi_shuffle = predictions.dimshuffle('x',0)
alf= T.stack([answer])

loss_ce = T.nnet.categorical_crossentropy(predictions.dimshuffle('x',0), T.stack([answer]))
print(loss_ce)


shuffler = theano.function(inputs = [predictions], outputs = [predi_shuffle])
trad = theano.function(inputs = [answer], outputs = [alf], allow_input_downcast = True)
loss_calculator = theano.function(inputs = [predictions, answer], outputs = [loss_ce])

pred = np.arange(5.0) +0.5
ans = 2
pred[2] = pred[2] +6


pred = pred/sum(pred)


print(np.shape(pred))
print(pred)
print(type(pred[0]))
print(type(pred))


shuffle = shuffler(pred)
print("-----------")
print(np.shape(shuffle))
print(shuffle)
print(type(shuffle))

t = trad(ans)
print(t)
print(np.shape(t))
print("------------")

loss = loss_calculator(pred, ans)
print(loss)
print(np.shape(loss))
print(type(loss))
chill_lost = loss[0]
print(chill_lost)
print(np.shape(chill_lost))
chiller_lost = chill_lost[0]
print(chiller_lost)
print(np.shape(chiller_lost))
exit()





inp = T.ivector()
inp2 = T.ivector()

def function(a):
    return a*a +3
    
def vect_add(a,b):
    return a+b
    
outputs, udpates = theano.scan(fn=vect_add, sequences = [inp, inp2], n_steps = 5)
added = outputs

outputs, updates = theano.scan(fn=function, sequences = inp, n_steps = 5)

out = outputs

lisa = inp.dimshuffle("x",0)

trainer = theano.function(inputs = [inp], outputs = [out], allow_input_downcast = True)
bob = theano.function(inputs = [inp], outputs = [lisa], allow_input_downcast = True)
aaa = theano.function(inputs = [inp, inp2], outputs = [added], allow_input_downcast = True)


zz = np.arange(5)
zzz = np.arange(5) + np.arange(5)

jack = trainer(zz)
bib = bob(zz)
blip = aaa(zz,zzz)

#print(jack)

print(blip)

print(bib)
print(type(bib[0]))
print(np.shape(bib[0]))
