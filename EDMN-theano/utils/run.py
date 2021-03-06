import os
import numpy as np
import utils
import time
import sklearn.metrics as metrics


def is_the_verb_only_different(answer, pred, ivocab):    
    for i in range(1, np.shape(answer)[0]+1):
        a = answer.pop(0)
        p = pred.pop(0)
        if(a!=p):
            if(i !=2):
                print(ivocab[a])
                print(ivocab[p])
                return 1
    return 0

def get_number_difference(answer, pred):
    dif = 0
    for i in range(1, np.shape(answer)[0]+1):
        a = answer.pop()
        p = pred.pop()
        if(a!=p):
            if(i!=2):
                dif = dif + 1
    return dif
    

def get_stat(dmn, vocab, nbr_stat):
    error = 0
    ivocab = dmn.ivocab
    for j in range(0, nbr_stat):
        step_data = dmn.step(j, 'minitest')
        answers = step_data["answers"]
        pred = step_data["multiple_prediction"]
        
        val_ans = []
        val_pred = []        
        
        for i in range(0,np.shape(pred)[1]):
                pred_temp = pred[:,i,:]
                for x in pred_temp.argmax(axis=1):
                    val_pred.append(x)
        answers = answers[0]
        for i in range(0,np.shape(answers)[0]):
            val_ans.append(answers[i])
        temp = get_number_difference(val_ans, val_pred)
        if(temp>0):
            error = error + 1
    acc = (nbr_stat-error)*100/nbr_stat
    print("Accuracy is",acc)

        

def do_minitest(dmn, vocab, multiple_ans=False,nbr_test=0, log_it = False, state_name = ""):
    #data = load_minitest(fname)
    
    
    ivocab = dmn.ivocab
    total_facts = []
    total_q = []
    total_ans = []
    total_pred = []  
    total_error = 0
    
    for j in range(0, nbr_test):
        y_true = []
        y_pred = []
        step_data = dmn.step(j,'minitest')
        answers = step_data["answers"]
        inputs = step_data["inputs"]
        question = step_data["question"]
        
        if(multiple_ans):
            ret_multiple = step_data["multiple_prediction"]
        else:
            prediction = step_data["prediction"]
        print("-------")
            
        
        w_input = []
        w_q = []
        print("==> reconstruction of input and question")
        for i in range(0, np.shape(inputs)[0]):
            w_input.append(utils.get_word(dmn.word2vec, inputs[i]))
        for i in range(0, np.shape(question)[0]):
            w_q.append(utils.get_word(dmn.word2vec, question[i]))           
        print("Facts:")
        print( ' '.join(w_input))
        print( ' '.join(w_q) + "?")
        total_facts.append(w_input)
        total_q.append(w_q)
        
        if(multiple_ans==False):
            print("==>Right answer is:")
            for x in answers:
                y_true.append(x)
                print(ivocab[x])
            print("==>Answer found by the model is:")
            for x in prediction.argmax(axis=1):
                y_pred.append(x)
                print(ivocab[x])
        else:
            print("==>Right answer is:")
            answers = answers[0]
            for i in range(0,np.shape(answers)[0]):
                ans = (ivocab[answers[i]])
                y_true.append(ans)
            print(' '.join(y_true) + ".")
            total_ans.append(y_true)
            
            print("==>Multiple answer found are:")
            list_pred = []
            for i in range(0,np.shape(ret_multiple)[1]):
                pred_temp = ret_multiple[:,i,:]
                for x in pred_temp.argmax(axis=1):
                    list_pred.append(ivocab[x])
            print(' '.join(list_pred) + '. ('+str(np.shape(ret_multiple)[1])+' answers)')
            total_pred.append(list_pred)
            total_error = total_error + get_number_difference(y_true, list_pred)
            
            if(log_it):
                if(state_name == ""):
                    raise Exception("Wrong or state_name. You must give the babi_id in order to log the results of the minitest")
                    
                infos = state_name.split(".")
               # info = states/dmn_multiple.h5.bs10.babi1.epoch5.test3.24979.state
                babi_temp = infos[4]
                epoch_temp = infos[5]
                id = babi_temp[4:]
                epoch_nbr = epoch_temp[5:]
                episode = {'C':total_facts, 'Q':total_q, 'A':total_ans,'AF':total_pred}
                babi_id = utils.babi_map[id]
                fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/minitest_log/en/%s_100k_log_%s.txt' % (babi_id, epoch_nbr))
                write_log_results(fname,episode)
    print("Error is",(total_error))
                

def write_log_results(fname, data):
    file_obj = open(fname, "w")
    total_f = data['C']
    total_q = data['Q']
    total_a = data['A']
    total_p = data['AF']
    for i in range(np.shape(total_f)[0]):
        facts = (' '.join(total_f[i]))
        question = (' '.join(total_q[i])) + "?"    
        RAnswer = (' '.join(total_a[i]))
        FAnswer = (' '.join(total_p[i]))
    
        file_obj.write("F:")
        file_obj.write(facts)
        file_obj.write("\nQ:")
        file_obj.write(question)
        file_obj.write("\nA:")
        file_obj.write(RAnswer)
        file_obj.write("\nP:")
        file_obj.write(FAnswer)
        file_obj.write("\n")
    
    
   
    
def extract_output_stats(prediction, answers):
    hard_acc = 1.0
    avg_acc = 0.0
    individual_acc = []
    for i in range(0, np.shape(prediction)[0]):
        if(prediction[i]!=answers[i]):
            hard_acc = 0
            individual_acc.append(0)
        else:
            individual_acc.append(1)
            avg_acc+=1.0
    return hard_acc, avg_acc/np.shape(prediction)[0], individual_acc
    
    
    


def do_epoch(args, dmn, mode, epoch, skipped=0, data_writer =""):
    '''
    :param mode: train or test mode are available
    :param epoch: number of epoch. Useful only for display and metadata purposes
    :param skipped: number of skipped epochs. Useful only for display and metadata purposes
    :Return avg_loss, skipped: Average loss for the epochs, and current number of skipped epochs
    '''
    
    y_true = []
    y_pred = []
    avg_loss = 0.0
    avg_acc = 0.0
    avg_hardacc = 0.0
    avg_softacc = 0.0
    avg_indiacc = np.zeros(dmn.answer_step_nbr)
    prev_time = time.time() 
    
    batches_per_epoch = dmn.get_batches_per_epoch(mode)
    
    for i in range(0, batches_per_epoch):
        step_data = dmn.step(i, mode)
        prediction = step_data["prediction"]
        if(dmn.type=="pointer"):
            pointers = step_data["pointers"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]
        current_skip = (step_data["skipped"] if "skipped" in step_data else 0)
        log = step_data["log"]
        
        skipped += current_skip
        
        
        if(dmn.type == "multiple"):
            val_ans = []
            val_pred = []        
            for j in range(0,np.shape(prediction)[1]):
                pred_temp = prediction[:,j,:]
                for x in pred_temp.argmax(axis=1):
                    val_pred.append(x)
                    
            answers = answers[0]
            for j in range(0,np.shape(answers)[0]):
                val_ans.append(answers[j])
            
            #error = get_number_difference(val_ans, val_pred)
            #print(error)
            
            hard_acc, soft_acc, indi_acc = extract_output_stats(val_pred, val_ans)
        
            #nbr_words = dmn.answer_step_nbr
            #current_acc = (nbr_words - error)/nbr_words
            #avg_acc += soft_acc
            avg_hardacc +=hard_acc
            avg_softacc += soft_acc
            avg_indiacc += indi_acc
            current_acc = soft_acc
            avg_acc = avg_softacc
        elif(dmn.type == "pointer"):
            hard_acc = 0
            if(pointers[0]==prediction[0] and pointers[1]==prediction[1]):
                hard_acc = 1
            acc_1 = float(1/(float(1 + abs(pointers[0] - prediction[0]))))
            acc_2 = float(1/(float(1 + abs(pointers[1] - prediction[1]))))
            current_acc = 0.5 * acc_1 + 0.5 * acc_2
            avg_acc += current_acc
            avg_hardacc += hard_acc
        else:
            current_acc = np.nan
        
        if current_skip == 0:
            avg_loss += current_loss
            
            for x in answers:
                if(dmn.type == "multiple"):
                    pass
                    #y_true.append(sum(x))
                elif(dmn.type == "pointer"):
                    y_true.append(pointers)
                else:
                    y_true.append(x)
            
            if(dmn.type == "multiple"):
                for x in prediction.argmax(axis=2):
                    y_pred.append(sum(x))
            elif(dmn.type == "pointer"):
                y_pred.append(prediction)
            else:
                for x in prediction.argmax(axis=1):
                    y_pred.append(x)
            
            # TODO: save the state sometimes
            if (i % args.log_every == 0):
                cur_time = time.time()
                #print ("  %sing: %d.%d / %d \t loss: %.3f, avg_loss: %.3f \t accuracy: %.3f, avg_acc: %.3f \t skipped: %d, %s \t time: %.2fs" % 
                print ("  %sing: %d.%d / %d \t loss: %.3f, avg_loss: %.3f \t acc: %.3f, avg_acc: %.3f, avg_hardacc: %.3f \t skipped: %d \t time: %.2fs" % 
                    (mode, epoch, i * args.batch_size, batches_per_epoch * args.batch_size, 
                     #current_loss, avg_loss / (i + 1), current_acc, avg_acc / (i + 1), skipped, log, cur_time - prev_time))
                     current_loss, avg_loss / (i + 1), current_acc, avg_acc / (i + 1), avg_hardacc/ (i + 1), skipped, cur_time - prev_time))

                prev_time = cur_time
                if(data_writer!=""):
                    line = str(str(epoch) + ", "+str(i*args.batch_size)+", "+str(current_loss)+", "+str(avg_loss/(i + 1))+", "+str(current_acc)+", "+
                    str(avg_acc/(i + 1))+", " + str(avg_hardacc/(i + 1)) + ", "+ str(avg_indiacc/(i + 1))+ "\n")
                    data_writer.write(line)
        if np.isnan(current_loss):
            print("==> current loss IS NaN. This should never happen :) ")
            exit()

    avg_loss /= batches_per_epoch
    print("\n  %s loss = %.5f" % (mode, avg_loss))
    print("confusion matrix:")
    #print(metrics.confusion_matrix(y_true, y_pred))
    accuracy = 0

    #accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    accuracy = 0
    print("accuracy: %.2f percent" % (accuracy * 100.0 / batches_per_epoch / args.batch_size))
    
    return avg_loss, skipped
0

