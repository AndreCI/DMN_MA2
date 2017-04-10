import random
import numpy as np

import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
from lasagne import layers
from lasagne import nonlinearities
import cPickle as pickle

from utils import utils
from utils import nn_utils

floatX = theano.config.floatX


class DMN_multiple:
    
    def __init__(self, babi_train_raw, babi_test_raw, word2vec, word_vector_size, 
                dim, mode, answer_module, answer_step_nbr, input_mask_mode, memory_hops, l2, 
                normalize_attention, **kwargs):
        '''
        Build the DMN
        :param babi_train_raw: train dataset
        :param babi_test_raw: test dataset
        :param word2vec: a dictionary containing the word embeddings TODO: Check if right
        :param word_vector_size: dimension of the word embeddings (50,100,200,300)
        :param dim: number of hidden units in input module GRU
        :param mode: train or test mode
        :param answer_module: answer module type: feedforward or recurrent
        :param input_mask_mode: input_mask_mode: word or sentence
        :param memory_hops: memory GRU steps
        :param l2: L2 regularization
        :param normalize_attention: enable softmax on attention vector
        :param **kwargs:
        '''

        print("==> not used params in DMN class:", kwargs.keys())
        self.vocab = {}
        self.ivocab = {}
        
        #save params
        self.word2vec = word2vec
        self.word_vector_size = word_vector_size
        self.dim = dim
        self.mode = mode
        self.answer_module = answer_module
        #TODO: add check of inputs
        if(answer_step_nbr<1):
            raise Exception('The number of answer step must be greater than 0')
        self.answer_step_nbr = answer_step_nbr
        self.input_mask_mode = input_mask_mode
        self.memory_hops = memory_hops
        self.l2 = l2
        self.normalize_attention = normalize_attention
        
        self.train_input, self.train_q, self.train_answer, self.train_input_mask = self._process_input(babi_train_raw)
        self.test_input, self.test_q, self.test_answer, self.test_input_mask = self._process_input(babi_test_raw)
        self.vocab_size = len(self.vocab)

        self.input_var = T.matrix('input_var')
        self.q_var = T.matrix('question_var')
        self.answer_var = T.matrix('answer_var')
        self.input_mask_var = T.ivector('input_mask_var')
        
            
        print("==> building input module")
        self.W_inp_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        #TODO why 3 different set of weights & bias?
        
        #This does some loop
        inp_c_history, _ = theano.scan(fn=self.input_gru_step, 
                    sequences=self.input_var,
                    outputs_info=T.zeros_like(self.b_inp_hid))
        
        self.inp_c = inp_c_history.take(self.input_mask_var, axis=0)
        
        #This seems to be the memory.
        self.q_q, _ = theano.scan(fn=self.input_gru_step, 
                    sequences=self.q_var,
                    outputs_info=T.zeros_like(self.b_inp_hid))

        self.q_q = self.q_q[-1] #take only last elem
        
        
        print("==> creating parameters for memory module")
        self.W_mem_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        #Attnetion mechanisms 2 layer FFNN weights & bias
        self.W_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.dim, 7 * self.dim + 2))
        self.W_2 = nn_utils.normal_param(std=0.1, shape=(1, self.dim))
        self.b_1 = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        self.b_2 = nn_utils.constant_param(value=0.0, shape=(1,))
        

        print("==> building episodic memory module (fixed number of steps: %d)" % self.memory_hops)
        memory = [self.q_q.copy()] #So q_q is memory initialization
        for iter in range(1, self.memory_hops + 1):
            current_episode = self.new_episode(memory[iter - 1])
            memory.append(nn_utils.GRU_update(memory[iter - 1], current_episode,
                                          self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                                          self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                                          self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid))
        
        last_mem = memory[-1]
        
        print("==> building answer module")
        self.W_a = nn_utils.normal_param(std=0.1, shape=(self.vocab_size, self.dim))
                
        if self.answer_module == 'recurrent':
            self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
            def answer_step(prev_a, prev_y):
                a = nn_utils.GRU_update(prev_a, T.concatenate([prev_y, self.q_q]),
                                  self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                                  self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                                  self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)
                
                y = nn_utils.softmax(T.dot(self.W_a, a))
                return [a, y]
            
            # TODO: add conditional ending
            dummy_ = theano.shared(np.zeros((self.vocab_size, ), dtype=floatX))
            results, updates = theano.scan(fn=answer_step,
                outputs_info=[last_mem, T.zeros_like(dummy_)],
                n_steps=self.answer_step_nbr)
                
            self.prediction = results[1][-1]
            self.multiple_predictions = results[1] #don't get the memory (i.e. a)
            
        
        else:
            raise Exception("invalid answer_module")
        
        
        print("==> collecting all parameters")
        self.params = [self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                  self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                  self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid,
                  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid,
                  self.W_b, self.W_1, self.W_2, self.b_1, self.b_2, self.W_a]
        
        if self.answer_module == 'recurrent':
            self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                              self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                              self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid]
        
        
        print("==> building loss layer and computing updates")
        #TODO: modify the loss
        def temp_loss(pred, ans, loss):
            return loss + T.nnet.categorical_crossentropy(pred,ans)
            
        loss_ce, updates = theano.scan(fn=temp_loss,
                                            sequences=[self.multiple_predictions,self.answer_var],
                                            n_steps=self.answer_step_nbr,
                                            outputs_info = [np.float64(0)])        
        
        #self.multiple_predictions = self.multiple_predictions.dimshuffle(1, 0)
        #self.loss_ce = T.nnet.categorical_crossentropy(self.multiple_predictions, self.answer_var).mean()
        self.loss_ce = loss_ce[0]
        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0
        
        self.loss = self.loss_ce + self.loss_l2
        
        updates = lasagne.updates.adadelta(self.loss, self.params)
        
        if self.mode == 'train':
            print("==> compiling train_fn")
            #TODO check if train funtcion is ok
            self.train_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.input_mask_var], 
                                       outputs=[self.multiple_predictions], 
                                                #self.loss], 
                                       #updates=updates, 
                                       allow_input_downcast = True,
                                       on_unused_input="warn")
        
        print("==> compiling test_fn")
        self.test_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.input_mask_var],
                                  outputs=[self.prediction, self.loss, self.inp_c, self.q_q, last_mem])
                                  
        print("==> compiling minitest_fn")
        self.minitest_fn = theano.function(inputs=[self.input_var, self.q_var,
                                                   self.input_mask_var],
                                                   outputs=[self.multiple_predictions])
                                  
        
        
       # if self.mode == 'train':
        #    print("==> computing gradients (for debugging)")
         #   gradient = T.grad(self.loss, self.params)
          #  self.get_gradient_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.input_mask_var], outputs=gradient)

    

    
    
    #This is some twisted implementations. I don't like it AT ALL.
    #TODO move or remove this shit
    def input_gru_step(self, x, prev_h):
        '''
        Call GRU_update with self parameters
        :param x: input for the GRU update
        :param prev_h: previous state
        :return next step for the input GRU
        '''
        return nn_utils.GRU_update(prev_h, x, self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                                     self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                                     self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid)
    
    
    #TODO Called only in new_episode. Move/remove it?
    #TODO prev_g seems useless.
    def new_attention_step(self, ct, prev_g, mem, q_q):
        '''
        Compute next g_t^i given c_t, m^i-1, q
        :param ct: facts representation
        :param prev_g: is useless
        :param mem: memory representation
        :param q_q: question represention
        :return G: the output of the simple FFNN
        '''
        cWq = T.stack([T.dot(T.dot(ct, self.W_b), q_q)])
        cWm = T.stack([T.dot(T.dot(ct, self.W_b), mem)])
        z = T.concatenate([ct, mem, q_q, ct * q_q, ct * mem, T.abs_(ct - q_q), T.abs_(ct - mem), cWq, cWm]) #direct from paper AMA:DMN for QA
        
        l_1 = T.dot(self.W_1, z) + self.b_1
        l_1 = T.tanh(l_1)
        l_2 = T.dot(self.W_2, l_1) + self.b_2
        G = T.nnet.sigmoid(l_2)[0]
        return G
        
        
    #TODO this is the modified GRU of the MemUpdate Mechanism.
    #Maybe move it elsewhere (as a child of a GRU class?)
    def new_episode_step(self, ct, g, prev_h):
        '''
        Compute the h_t^i for the MemUpdate Mechanism
        :param ct: facts representation
        :param g: weights of the gates g^i (given by the attention mechanism)
        :param prev_h: previous state of the Mem GRU (h_t-1^i)
        :return h_t^i: next state
        '''
        gru = nn_utils.GRU_update(prev_h, ct,
                             self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                             self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                             self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid)
        
        h = g * gru + (1 - g) * prev_h
        return h
       
    
    def new_episode(self, mem):
        '''
        Create a new episode
        Compute the g using the attention mechanism
        Compute the new state h_t^i of the mem update mechanism
        Use it to compute e^i = h_T_C^i (see paper AMA:DMN for QA)
        :param mem: current memory
        :return e[-1]: latest episode
        '''
        #g_updates seems useless.
        g, g_updates = theano.scan(fn=self.new_attention_step,
            sequences=self.inp_c,
            non_sequences=[mem, self.q_q],
            outputs_info=T.zeros_like(self.inp_c[0][0])) 
        
        #Softmax if normalize_attention?
        if (self.normalize_attention):
            g = nn_utils.softmax(g)
        
        #e_updates seems useless.
        e, e_updates = theano.scan(fn=self.new_episode_step,
            sequences=[self.inp_c, g],
            outputs_info=T.zeros_like(self.inp_c[0]))
        
        return e[-1]


    #TODO add documentation (later because it isn't really useful)    
    def save_params(self, file_name, epoch, **kwargs):
        '''
        Basic function to save current state.
        '''
        with open(file_name, 'w') as save_file:
            pickle.dump(
                obj = {
                    'params' : [x.get_value() for x in self.params],
                    'epoch' : epoch, 
                    'gradient_value': (kwargs['gradient_value'] if 'gradient_value' in kwargs else 0)
                },
                file = save_file,
                protocol = -1
            )
    
    #TODO add documentation (later because it isn't really useful)    
    def load_state(self, file_name):
        '''
        Basic function to load an old state
        '''
        print("==> loading state %s" % file_name)
        with open(file_name, 'r') as load_file:
            dict = pickle.load(load_file)
            loaded_params = dict['params']
            for (x, y) in zip(self.params, loaded_params):
                x.set_value(y)

    
    def _process_input(self, data_raw):
        '''
        :param data_raw: raw data (train or test set) from outils.get_babi_raw
        :return inputs: all the inputs, as a list of word vector representation. 
        :return questions: all the questions, as a list of word vector repre.
        :return answers: all the answers, as a list of word vec repre
        :return input_masks:
        '''
        questions = []
        inputs = []
        answers = []
        input_masks = []
        for x in data_raw:
            inp = x["C"].lower().split(' ') 
            inp = [w for w in inp if len(w) > 0]
            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]
            ans = x["A"].lower().split(' ')
            ans = [w for w in ans if len(w) > 0]
            
            inp_vector = [utils.process_word(word = w, 
                                        word2vec = self.word2vec, 
                                        vocab = self.vocab, 
                                        ivocab = self.ivocab, 
                                        word_vector_size = self.word_vector_size, 
                                        to_return = "word2vec") for w in inp] #for each word, get the word vec rpz
                                        
            q_vector = [utils.process_word(word = w, 
                                        word2vec = self.word2vec, 
                                        vocab = self.vocab, 
                                        ivocab = self.ivocab, 
                                        word_vector_size = self.word_vector_size, 
                                        to_return = "word2vec") for w in q]
            
            inputs.append(np.vstack(inp_vector).astype(floatX))
            questions.append(np.vstack(q_vector).astype(floatX))
            
            ans_vector = [utils.process_word(word = w,
                                             word2vec = self.word2vec,
                                             vocab = self.vocab,
                                             ivocab = self.ivocab,
                                             word_vector_size = self.word_vector_size,
                                             to_return = "word2vec") for w in ans]
            ans_vector = ans_vector[0:len(ans_vector)-1]
            answers.append(np.vstack(ans_vector).astype(floatX))                                 
                                             
#            answers.append(utils.process_word(word = x["A"], 
#                                            word2vec = self.word2vec, 
#                                            vocab = self.vocab, 
#                                            ivocab = self.ivocab, 
#                                            word_vector_size = self.word_vector_size, 
#                                            to_return = "index"))
            # NOTE: here we assume the answer is one word!
            #TODO check what the heck input_masks is made of.
            if self.input_mask_mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) 
            elif self.input_mask_mode == 'sentence': 
                input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32)) 
            else:
                raise Exception("invalid input_mask_mode") #TODO this should probably not be raised here... 
        
        return inputs, questions, answers, input_masks

    
    def get_batches_per_epoch(self, mode):
        if (mode == 'train'):
            return len(self.train_input)
        elif (mode == 'test'):
            return len(self.test_input)
        else:
            raise Exception("unknown mode")
    
    
    def shuffle_train_set(self):
        print("==> Shuffling the train set")
        combined = zip(self.train_input, self.train_q, self.train_answer, self.train_input_mask)
        random.shuffle(combined)
        self.train_input, self.train_q, self.train_answer, self.train_input_mask = zip(*combined)
        
    
    def step(self, batch_index, mode):
        '''
        
        :param batch_index:
        :param mode: train or test
        :return a directory containing:
            :prediction:
            :answers:
            :current_loss:
            :skipped:
            :log:
        '''
        if mode == "train" and self.mode == "test":
            raise Exception("Cannot train during test mode")
        
        if mode == "train":
            print("TRAIN (431)")
            theano_fn = self.train_fn 
            inputs = self.train_input
            qs = self.train_q
            answers = self.train_answer
            input_masks = self.train_input_mask
        elif mode == "test":    
            theano_fn = self.test_fn 
            inputs = self.test_input
            qs = self.test_q
            answers = self.test_answer
            input_masks = self.test_input_mask
        elif mode == "minitest":    
            theano_fn = self.test_fn 
            theano_fn2 = self.minitest_fn
            inputs = self.test_input
            qs = self.test_q
            answers = self.test_answer
            input_masks = self.test_input_mask
        else:
            raise Exception("Invalid mode")
            
        inp = inputs[batch_index]
        q = qs[batch_index]
        ans = answers[batch_index]
        input_mask = input_masks[batch_index]

        skipped = 0
        grad_norm = float('NaN')
        
      #  if mode == 'train':
       #     gradient_value = self.get_gradient_fn(inp, q, ans, input_mask)
        #    grad_norm = np.max([utils.get_norm(x) for x in gradient_value])
            
          #   if (np.isnan(grad_norm)):
           #     print("==> gradient is nan at index %d." % batch_index)
            #    print("==> skipping")
             #   skipped = 1
        
        if skipped == 0:      
            print(np.shape(ans))
            
            ret = theano_fn(inp, q, ans, input_mask)
            print(np.shape(ret[0]))
            print(np.shape(ret))
        
            
            print("------------------------")
            print("COMPILATION SUCCESSFUL")
            print("Stopping now to gain computation time.")
            return exit()
            ret_multiple = theano_fn2(inp,q,input_mask, ans)
        else:
            ret = [-1, -1]
        ret = ret_multiple
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        return {"inputs":inp,
                "question":q,
                "prediction": np.array([ret[0]]),
                "multiple_prediction": np.array(ret_multiple),
                "answers": np.array([ans]),
                "current_loss": ret[0],
                "skipped": skipped,
                "log": "pn: %.3f \t gn: %.3f" % (param_norm, grad_norm)
                }
        