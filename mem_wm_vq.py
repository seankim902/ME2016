# -*- coding: utf-8 -*-


import os
import time
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
import tensorflow as tf



def prepare_data(name, video_path, wav_path, maxlen=76, dim_wav = 5512, dim_img = 4096):
    
    video_nm = name.apply(lambda x : os.path.join(video_path, x+ '.npy'))
    wav_nm = name.apply(lambda x : os.path.join(wav_path, x+ '.npy'))
    
    n_samples = len(name)

    video_x = map(lambda x: np.load(x), video_nm)
    video_x = np.array(video_x)
    wav_seqs_x = map(lambda x: np.load(x), wav_nm) # sample * step * dim_wav
    wav_x = np.zeros((n_samples, maxlen, dim_wav))
    for idx, s_x in enumerate(wav_seqs_x):  
        wav_x[idx][:len(s_x)] = s_x

    return video_x, wav_x
    

def get_minibatch_indices(n, batch_size, shuffle=False):

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size
    if (minibatch_start != n):   
        minibatches.append(idx_list[minibatch_start:])
    return minibatches
    
    
    
class MemN2N(object):
    
    def __init__(self, config):
        self.dim_img = config.dim_img
        self.dim_wav = config.dim_wav
        self.dim_mem = config.dim_mem
        self.mem_size = self.steps = config.mem_size
        
        self.n_hop = config.n_hop
        self.batch_size = config.batch_size
        self.do_prob = config.do_prob # keep prob
        self.nl = config.nl # keep prob
        self.learning_rate = config.learning_rate

        self.global_step = tf.Variable(0, name='g_step')
        self.A = tf.Variable(tf.random_normal([self.dim_wav, self.dim_mem]), name='A')
        self.b_A = tf.Variable(tf.random_normal([self.dim_mem]), name='b_A')
        self.B = tf.Variable(tf.random_normal([self.dim_img, self.dim_mem]), name='B')
        self.b_B = tf.Variable(tf.random_normal([self.dim_mem]), name='b_B')
        self.C = tf.Variable(tf.random_normal([self.dim_wav, self.dim_mem]), name='C')
        self.b_C = tf.Variable(tf.random_normal([self.dim_mem]), name='b_C')
        
        self._temporal = tf.linspace(0.0, np.float32(self.mem_size-1), self.mem_size)
        self.T_A = self.T_C = tf.Variable(self._temporal/tf.reduce_sum(self._temporal))
        
        self.W_o = tf.Variable(tf.random_normal([self.dim_mem, 1]))
        self.b_o = tf.Variable(tf.random_normal([1]))
        


    def build_model(self):
        
        m_auditory = tf.placeholder(tf.float32, [self.batch_size, self.mem_size, self.dim_wav], name="m")
        q_visual = tf.placeholder(tf.float32, [self.batch_size, self.dim_img], name="q")
        y = tf.placeholder(tf.float32, [self.batch_size, 1], name="y")


        m_ = tf.reshape(m_auditory, [-1, self.dim_wav])
        # sample * mem_size * dim -> (sample * mem_size) * dim
        m_input = tf.nn.xw_plus_b(m_, self.A, self.b_A)
        m_input = tf.reshape(m_input, [self.batch_size, self.mem_size, self.dim_mem])
        m_input = tf.transpose(m_input, perm=[0, 2, 1]) + self.T_A #  sample * dim_mem * mem_size
        m_input = tf.transpose(m_input, perm=[0, 2, 1]) #  sample * mem_size * dim_mem

        m_output = tf.nn.xw_plus_b(m_, self.C, self.b_C)
        m_output = tf.reshape(m_output, [self.batch_size, self.mem_size, self.dim_mem])
        m_output = tf.transpose(m_output, perm=[0, 2, 1]) + self.T_C #  sample * dim_mem * mem_size
        m_output = tf.transpose(m_output, perm=[0, 2, 1]) #  sample * mem_size * dim_mem
        
        u = tf.nn.xw_plus_b(q_visual, self.B, self.b_B)
        os = [] 
        for _ in range(self.n_hop):
            u = tf.expand_dims(u, -1) # sample * dim_mem -> sample * dim_mem * 1
            m_prob = tf.batch_matmul(m_input, u)
            m_prob = tf.squeeze(m_prob)
            m_prob = tf.nn.softmax(m_prob) # sample * mem_size
    
            weighted_m = tf.mul(m_output, tf.tile(tf.expand_dims(m_prob, 2), [1, 1, self.dim_mem]))
            o = tf.reduce_sum(weighted_m, 1) # sample * dim_mem
            
            u = tf.add(tf.squeeze(u), o)


            if self.nl:
                u = tf.nn.elu(u)
            
            os.append(u)

        train_hidden = valid_hidden = os[-1]
        
        train_hidden = tf.nn.dropout(train_hidden, self.do_prob)
        y_hat = tf.nn.xw_plus_b(train_hidden, self.W_o, self.b_o)
        v_y_hat = tf.nn.xw_plus_b(valid_hidden, self.W_o, self.b_o)

        loss = tf.reduce_mean(tf.square(y_hat - y))
        regularizers = (tf.nn.l2_loss(self.A) + tf.nn.l2_loss(self.b_A) +
                        tf.nn.l2_loss(self.B) + tf.nn.l2_loss(self.b_B) +
                        tf.nn.l2_loss(self.C) + tf.nn.l2_loss(self.b_C) +
                        tf.nn.l2_loss(self.T_A) + tf.nn.l2_loss(self.T_C) +
                        tf.nn.l2_loss(self.W_o) + tf.nn.l2_loss(self.b_o))        
        loss += 5e-3 * regularizers
        
        
        lr = tf.train.exponential_decay( self.learning_rate,  # Base learning rate.
                                         self.global_step,         # Current index
                                         28,                 # Decay step.
                                         0.96,                # Decay rate.
                                         staircase=True)
        
      
        
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=self.global_step)
        
        
        
        return m_auditory, q_visual, y, loss, train_op, lr, v_y_hat




def train():
    config = get_config()

    data = pd.read_pickle('/home/devbox/me2016/task1/data.pkl')

    wav_path = '/home/devbox/me2016/task1/auditory_m'
    video_path = '/home/devbox/me2016/task1/visual_q'    
    
    
    train = data[data['set']!=0]
    valid = data[data['set']==0]

    train_nm = train['name']
    train_y_v = train['valenceValue']
    train_y_a = train['arousalValue']
    valid_nm = valid['name']
    valid_y_v = valid['valenceValue']
    valid_y_a = valid['arousalValue']
    


    train_y = train_y_v
    valid_y = valid_y_v
    valid_y = np.array(valid_y)[:,None]
    train_y = np.array(train_y)[:,None]
    valid_video_x, valid_wav_x = prepare_data(valid_nm, video_path, wav_path, config.mem_size, config.dim_wav, config.dim_img)
    valid_batch_indices=get_minibatch_indices(len(valid_video_x), config.batch_size, shuffle=False)

    train_video_x, train_wav_x = prepare_data(train_nm, video_path, wav_path, config.mem_size, config.dim_wav, config.dim_img)

    
    print 'valid_video_x :', valid_video_x.shape
    print 'valid_wav_x :', valid_wav_x.shape
    print 'valid_y :',valid_y.shape


    
    with tf.Session() as sess:
        initializer = tf.random_normal_initializer(0, 0.01)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = MemN2N(config = config)          
            
        m_auditory, q_visual, y, loss, train_op, lr, v_y_hat = model.build_model()
        
        
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        #saver.restore(sess, config.model_ckpt_path+'-41')
        
        best = np.Inf
        for i in range(config.epoch):
            start = time.time()
            
            
            batch_indices=get_minibatch_indices(len(train_video_x), config.batch_size, shuffle=True)
    
            
            for j, indices in enumerate(batch_indices):
                m_ = np.array([ train_wav_x[k,:,:] for k in indices])
                q_ = np.array([ train_video_x[k,:] for k in indices])
                y_ = np.array([ train_y[k,:] for k in indices])
                
                
                _loss,  _, _lr  = sess.run([loss, train_op, lr],
                                              {m_auditory : m_,
                                               q_visual : q_,
                                               y: y_})
                if j % 4 == 0 :
                    print 'cost : ', _loss, ', lr : ', _lr, ', iter : ', j+1, ' in epoch : ',i+1
            print 'cost : ', _loss, ', lr : ', _lr, ', iter : ', j+1, ' in epoch : ',i+1,' elapsed time : ', int(time.time()-start)
            
            if config.valid_epoch is not None:  # for validation
                
                
                if (i+1) % config.valid_epoch == 0:
                    val_preds = []
                    for j, indices in enumerate(valid_batch_indices):
                        m_ = np.array([ valid_wav_x[k,:,:] for k in indices])
                        q_ = np.array([ valid_video_x[k,:] for k in indices])
                        y_ = np.array([ valid_y[k,:] for k in indices])
                        
                        _pred = sess.run(v_y_hat,
                                        {m_auditory : m_,
                                         q_visual : q_,
                                         y: y_})
               
                        val_preds = val_preds + _pred.tolist()
                    
                    valid_loss = mean_squared_error(valid_y, val_preds)
                    print '[', best, '] ##### valid loss : ', valid_loss, ' after epoch ', i+1
                    if valid_loss < best :
                        best = valid_loss
                        print 'save model...',
                        saver.save(sess, config.model_ckpt_path, global_step=int(best))
                        print int(best)
        
        print 'best valid accuracy :', best
                                       


def get_config():
    class Config1(object):
        
        dim_img = 4096
        dim_wav = 5512
        dim_mem = 256
        mem_size = steps = 76 # 19 * 4
        
        n_hop = 2        
        batch_size = 350
        do_prob = 0.6
        nl = True
        learning_rate = 0.01
        epoch = 200
        valid_epoch = 1
        model_ckpt_path = '/home/devbox/workspace/seonhoon/mem_model.ckpt'
        
        
    return Config1()
    
    
def main(_):


    is_train = True  # if False then test
    
    if is_train :
        train()
    

if __name__ == "__main__":
  tf.app.run()