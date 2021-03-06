import random
import numpy as np

import theano
from theano.ifelse import ifelse
import theano.tensor as T

import lasagne
import cPickle as pickle

from utils import utils
from utils import nn_utils

floatX = theano.config.floatX


class DMN_pointer:
    
    def __init__(self, babi_train_raw, babi_test_raw, word2vec, word_vector_size, 
                dim, mode, answer_module, answer_step_nbr, input_mask_mode, memory_hops, l2, 
                normalize_attention, max_input_size, **kwargs):
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
        self.type = "pointer"
        self.vocab = {}
        self.ivocab = {}
        
        print(max_input_size)        
        
        #save params
        self.word2vec = word2vec
        self.word_vector_size = word_vector_size
        self.dim = dim #number of hidden units in input layer GRU
        self.pointer_dim = max_input_size #maximal size for the input, used as hyperparameter
        self.mode = mode
        self.answer_module = answer_module
        self.answer_step_nbr = answer_step_nbr
        self.input_mask_mode = input_mask_mode
        self.memory_hops = memory_hops
        self.l2 = l2
        self.normalize_attention = normalize_attention
        
        self.train_input, self.train_q, self.train_answer, self.train_input_mask, self.train_pointers_s, self.train_pointers_e = self._process_input(babi_train_raw)
        self.test_input, self.test_q, self.test_answer, self.test_input_mask, self.test_pointers_s, self.test_pointers_e = self._process_input(babi_test_raw)
        self.vocab_size = len(self.vocab)

        self.input_var = T.matrix('input_var')
        self.q_var = T.matrix('question_var')
        self.answer_var = T.ivector('answer_var')
        self.input_mask_var = T.ivector('input_mask_var')
        self.pointers_s_var = T.ivector('pointers_s_var')
        self.pointers_e_var = T.ivector('pointer_e_var')
        
            
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
        
        #in case of multiple sentences, only keep the hidden states which index match the <eos> char
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
        
        self.last_mem = memory[-1]
        
                
        
        print("==> building answer module")
        self.Ws_p = nn_utils.normal_param(std=0.1, shape=(self.pointer_dim, self.dim)) #shape must be size_input * mem_size = self.dim
        self.We_p = nn_utils.normal_param(std=0.1, shape=(self.pointer_dim, self.dim))
        self.Wh_p = nn_utils.normal_param(std=0.1, shape=(self.pointer_dim, self.dim))
        self.Ws_pr = nn_utils.normal_param(std=0.1, shape=(self.pointer_dim, self.dim)) #shape must be size_input * mem_size = self.dim
        self.We_pr = nn_utils.normal_param(std=0.1, shape=(self.pointer_dim, self.dim))
        self.Wh_pr = nn_utils.normal_param(std=0.1, shape=(self.pointer_dim, self.dim))
        
        self.Psp = nn_utils.softmax(T.dot(self.Ws_p, self.last_mem)) #size must be == size_input
        self.Pepr = nn_utils.softmax(T.dot(self.We_pr, self.last_mem))
        
        #TODO:
        self.start_idx = T.argmax(self.Psp)
        self.end_idxr = T.argmax(self.Pepr)
        
        self.start_idx_state = inp_c_history[self.start_idx] #must be hidden state idx idx_max_val(Psp)  self.last_mem#
        self.end_idx_state = inp_c_history[self.end_idxr]
        #temp1 = T.dot(self.We_p, self.last_mem)
        #temp2 = T.dot(self.Wh_p, self.start_idx_state)
        #temp3 = temp1 + temp2
        self.Pep = nn_utils.softmax(T.dot(self.We_p, self.last_mem) + T.dot(self.Wh_p, self.start_idx_state)) #size must be == size_input
        self.Pspr = nn_utils.softmax(T.dot(self.Ws_pr, self.last_mem) + T.dot(self.Wh_pr, self.end_idx_state))
        
        Ps = (self.Psp + self.Pspr)/2
        Pe = (self.Pep + self.Pepr)/2
        self.start_idxr = T.argmax(self.Pspr)
        self.end_idx = T.argmax(self.Pep)   
        
        self.start_idx_f = T.argmax(Ps)#(self.start_idx + self.start_idxr)/2
        self.end_idx_f = T.argmax(Pe)#(self.end_idx + self.end_idxr)/2
        
        #multiple_answers = []
        #bboole = T.lt(self.start_idx_f, self.end_idx_f)
                
        
        #trange = ifelse(bboole, T.arange(self.start_idx_f, self.end_idx_f), T.arange(self.start_idx_f - 1, self.start_idx_f))
        
        
#        self.W_a = nn_utils.normal_param(std=0.1, shape=(self.vocab_size, self.dim))
#                
#        if self.answer_module == 'recurrent':
#            self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.dim + self.vocab_size))
#            self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
#            self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
#            
#            self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.dim + self.vocab_size))
#            self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
#            self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
#            
#            self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.dim + self.vocab_size))
#            self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
#            self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
#        
#            def answer_step(prev_a, prev_y):
#                a = nn_utils.GRU_update(prev_a, T.concatenate([prev_y, self.q_q, self.last_mem]),
#                                  self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
#                                  self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
#                                  self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)
#                
#                y = nn_utils.softmax(T.dot(self.W_a, a))
#                return [a, y]
#            
#            # TODO: add conditional ending
#            dummy_ = theano.shared(np.zeros((self.vocab_size, ), dtype=floatX))
#            results, updates = theano.scan(fn=answer_step,
#                outputs_info=[self.last_mem, T.zeros_like(dummy_)],
#                n_steps=self.answer_step_nbr)
#                
#            self.multiple_predictions = results[1] #don't get the memory (i.e. a)
#            
#        
#        else:
#            raise Exception("invalid answer_module")
        
        
        print("==> collecting all parameters")
        self.params = [self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                  self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                  self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid,
                  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid,
                  self.W_b, self.W_1, self.W_2, self.b_1, self.b_2, self.Ws_p, self.We_p, self.Wh_p,
                  self.Ws_pr, self.We_pr, self.Wh_pr]
        
#        if self.answer_module == 'recurrent':
#            self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
#                              self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
#                              self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid]
#        
#        
        print("==> building loss layer and computing updates")
#        def temp_loss(curr_pred, curr_ans, loss):
#            temp = T.nnet.categorical_crossentropy(curr_pred.dimshuffle("x",0),T.stack([curr_ans]))[0]
#            return loss + temp
#            
#        outputs, updates = theano.scan(fn=temp_loss,
#                                            sequences=[self.multiple_predictions, self.answer_var],
#                                            outputs_info = [np.float64(0.0)],
#                                            n_steps=self.answer_step_nbr)        
        
        
        loss_start = T.nnet.categorical_crossentropy(Ps.dimshuffle("x",0), T.stack([self.pointers_s_var[0]]))[0]
        loss_end = T.nnet.categorical_crossentropy(Pe.dimshuffle("x",0), T.stack([self.pointers_e_var[0]]))[0]
        #loss_1 = Ps
#        def temp_loss(curr_idx, curr_ans, loss):
#            curr_pred = self.input_var[curr_idx]
#            temp = T.nnet.catergorical_crossentropy(curr_pred, curr_ans)[0]
#            return loss + temp
#            
#        outputs, udpates = theano.scan(fn=temp_loss,
#                                       sequences = [answers_range, self.answer_var],
#                                        outputs_info = [np.float64(0.0)],
#                                        n_steps = ???)        
       
#        self.loss_ce = outputs[-1]
        #temp1 = (self.end_idx_f - self.pointers_e_var)
        #temp2 = T.abs_(temp1) #* temp1
        #temp3 = (self.start_idx_f)# - self.pointers_s_var)
        #temp4 = T.abs_(temp3) #* temp3
        self.loss_ce = loss_start + loss_end #(temp2 + temp4)
        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0
        
        self.loss = self.loss_ce + self.loss_l2
        
        updates = lasagne.updates.adadelta(self.loss, self.params)
        
        if self.mode == 'train':
            print("==> compiling train_fn")
            self.train_fn = theano.function(inputs=[self.input_var, self.q_var, self.input_mask_var, self.pointers_s_var, self.pointers_e_var], 
                                       outputs=[self.start_idx_f, 
                                                self.end_idx_f,
                                                self.loss], 
                                       updates=updates,
                                       allow_input_downcast = True)
        if self.mode != 'minitest':
            print("==> compiling test_fn")
            self.test_fn = theano.function(inputs=[self.input_var, self.q_var, self.input_mask_var, self.pointers_s_var, self.pointers_e_var],
                                                   outputs=[self.start_idx_f, self.end_idx_f, self.loss, self.inp_c, self.q_q],
                                      allow_input_downcast = True)
        
        if self.mode == 'minitest':                          
            print("==> compiling minitest_fn")
            self.minitest_fn = theano.function(inputs=[self.input_var, self.q_var,
                                                       self.input_mask_var, self.pointers_s_var, self.pointers_e_var],
                                                       outputs=[self.start_idx_f, self.end_idx_f])                                  
    
    
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
       
    
    def new_episode(self, mem, all_h=False):
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
        pointers_s = []
        pointers_e = []
        for x in data_raw:
            inp = x["C"].lower().split(' ') 
            normal_len = np.shape(inp)[0]
            inp = [w for w in inp if len(w) > 0]
            while(np.shape(inp)[0] < normal_len):
                inp.append(" <eoc>")
            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]
            ans = x["A"].lower().split(' ')
            ans = [w for w in ans if len(w) > 0]
            
            pointers_s.append(x["Ps"])
            pointers_e.append(x["Pe"])
            
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
                                             to_return = "index") for w in ans]
                                   
            ans_vector = ans_vector[0:len(ans_vector)-1]
            answers.append(np.vstack(ans_vector).astype(floatX))                                 
            
            #print(np.shape(inp_vector))            
            
                                             
            if self.input_mask_mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) 
            elif self.input_mask_mode == 'sentence': 
                #Mask is here an array containing the index of '.'
                input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32)) 
            else:
                raise Exception("invalid input_mask_mode") #TODO this should probably not be raised here... 
        

        return inputs, questions, answers, input_masks, pointers_s, pointers_e

    
    def get_batches_per_epoch(self, mode):
        if (mode == 'train'):
            return len(self.train_input)
        elif (mode == 'test'):
            return len(self.test_input)
        else:
            raise Exception("unknown mode")
    
    
    def shuffle_train_set(self):
        print("==> Shuffling the train set")
        combined = zip(self.train_input, self.train_q, self.train_answer, self.train_input_mask, self.train_pointers_s, self.train_pointers_e)
        random.shuffle(combined)
        self.train_input, self.train_q, self.train_answer, self.train_input_mask, self.train_pointers_s, self.train_pointers_e = zip(*combined)
        
    
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
            theano_fn = self.train_fn 
            inputs = self.train_input
            qs = self.train_q
            answers = self.train_answer
            input_masks = self.train_input_mask
            pointers_s = self.train_pointers_s
            pointers_e = self.train_pointers_e
        elif mode == "test":    
            theano_fn = self.test_fn 
            inputs = self.test_input
            qs = self.test_q
            answers = self.test_answer
            input_masks = self.test_input_mask
            pointers_s = self.train_pointers_s
            pointers_e = self.train_pointers_e
        elif mode == "minitest":    
            theano_fn = self.minitest_fn
            inputs = self.test_input
            qs = self.test_q
            answers = self.test_answer
            input_masks = self.test_input_mask
            pointers_s = self.train_pointers_s
            pointers_e = self.train_pointers_e
        else:
            raise Exception("Invalid mode")
            

        inp = inputs[batch_index]
        q = qs[batch_index]
        ans = answers[batch_index]
        ans = ans[:,0] #reshape from (5,1) to (5,)
        input_mask = input_masks[batch_index]
        pointer_s = pointers_s[batch_index]
        pointer_e = pointers_e[batch_index]

        skipped = 0
        grad_norm = float('NaN')
        
        
        if skipped == 0:      
            
            #Answer MUST(?) be a vector containing number corresponding to the words in ivocab. i.e. [1, 8, 3, 9, 14] (=[5])
            #MulPread must be a vector containing probabilities for each words in vocab, i.e. [5*dic_size] (=[5*20] usually)          
            
            pointer_s = np.stack([pointer_s])
            pointer_e = np.stack([pointer_e])
            
            if(mode == "minitest"):
                ret_multiple = theano_fn(inp, q, input_mask, pointer_s, pointer_e)
            else:
                ret_multiple = theano_fn(inp, q, input_mask, (pointer_s), pointer_e)
            
        else:
            ret_multiple = [-1, -1]
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        if mode != 'minitest':
            return {"inputs":inp,
                    "question":q,
                    "prediction": np.array([ret_multiple[0], ret_multiple[1]]),
                    "answers": np.array([ans]),
                    "pointers":np.array([pointer_s, pointer_e]),
                    "current_loss": ret_multiple[2],
                    "skipped": skipped,
                    "log": "pn: %.3f \t gn: %.3f" % (param_norm, grad_norm)
                    }
        else:
            return {"inputs":inp,
                    "question":q,
                    "multiple_prediction": np.array(ret_multiple),
                    "answers": np.array([ans]),
                    "skipped": skipped,
                    "log": "pn: %.3f \t gn: %.3f" % (param_norm, grad_norm)
                    }
        