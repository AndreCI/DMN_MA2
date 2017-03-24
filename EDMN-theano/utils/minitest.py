import numpy as np
import utils

def do_minitest(dmn, vocab):
    #data = load_minitest(fname)
    
    y_true = []
    y_pred = []
    loss = 0.0
    ivocab = dmn.ivocab
    step_data = dmn.step(1,'test')
    prediction = step_data["prediction"]
    answers = step_data["answers"]
    inputs = step_data["inputs"]
    question = step_data["question"]
    
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
    print("==>Right answer is:")
    for x in answers:
        y_true.append(x)
        print(ivocab[x])
    print("==>Answer found by the model is:")
    for x in prediction.argmax(axis=1):
        y_pred.append(x)
        print(ivocab[x])

    


def do_epoch(mode, epoch, skipped=0):
    '''
    :param mode: train or test mode are available
    :param epoch: number of epoch. Useful only for display and metadata purposes
    :param skipped: number of skipped epochs. Useful only for display and metadata purposes
    :Return avg_loss, skipped: Average loss for the epochs, and current number of skipped epochs
    '''
    y_true = []
    y_pred = []
    avg_loss = 0.0
    prev_time = time.time() 
    
    batches_per_epoch = dmn.get_batches_per_epoch(mode)
    
    for i in range(0, batches_per_epoch):
        step_data = dmn.step(i, mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]
        current_skip = (step_data["skipped"] if "skipped" in step_data else 0)
        log = step_data["log"]
        
        skipped += current_skip
        
        if current_skip == 0:
            avg_loss += current_loss
            
            for x in answers:
                y_true.append(x)
            
            for x in prediction.argmax(axis=1):
                y_pred.append(x)
            
            # TODO: save the state sometimes
            if (i % args.log_every == 0):
                cur_time = time.time()
                print ("  %sing: %d.%d / %d \t loss: %.3f \t avg_loss: %.3f \t skipped: %d \t %s \t time: %.2fs" % 
                    (mode, epoch, i * args.batch_size, batches_per_epoch * args.batch_size, 
                     current_loss, avg_loss / (i + 1), skipped, log, cur_time - prev_time))
                prev_time = cur_time
        
        if np.isnan(current_loss):
            print("==> current loss IS NaN. This should never happen :) ")
            exit()

    avg_loss /= batches_per_epoch
    print("\n  %s loss = %.5f" % (mode, avg_loss))
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_true, y_pred))
    
    accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    print("accuracy: %.2f percent" % (accuracy * 100.0 / batches_per_epoch / args.batch_size))
    
    return avg_loss, skipped