from collections import OrderedDict
from typing import Sequence, Any
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import optparse


from dyPred import Model
from utils import glorot_init

def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=True):
    bwd_edge_offset = 0 if tie_fwd_bkwd else (num_edge_types // 2)
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
    target = []
    for src, e, dest in graph:
        amat[e-1, dest, src] = 1
        amat[e-1 + bwd_edge_offset, src, dest] = 1
    return amat

def decoder_output(amat):
    target = []

    for i in range(len(amat[0])):
        for j in range(len(amat[0][i])):
            target.append(amat[0][i][j])
    target=np.asarray(target,dtype=np.float32)
    return target



class GGNNModel(Model):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
                        'dropout_keep_prob': 1.,
                        'task_sample_ratios': {},
                        'use_edge_bias': True
                      })
        return params

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        # inputs
        self.placeholders['dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='dropout_keep_prob')
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32,
                                                                          [None, None, self.params['hidden_size']],
                                                                          name='node_features')
        self.placeholders['node_mask'] = tf.placeholder(tf.float32, [None, None], name='node_mask')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, ())
        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32,
                                                               [None, self.num_edge_types, None, None])
        self.__adjacency_matrix = tf.transpose(self.placeholders['adjacency_matrix'], [1, 0, 2, 3])


        # weights
        self.weights['edge_weights'] = tf.Variable(glorot_init([self.num_edge_types, h_dim, h_dim]))
        if self.params['use_edge_bias']:
            self.weights['edge_biases'] = tf.Variable(np.zeros([self.num_edge_types, 1, h_dim]).astype(np.float32))
        with tf.variable_scope("gru_scope"):
            cell = tf.contrib.rnn.GRUCell(h_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 state_keep_prob=self.placeholders['dropout_keep_prob'])
            self.weights['node_gru'] = cell

    def compute_final_node_representations(self) -> tf.Tensor:
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        h = self.placeholders['initial_node_representation']
        h = tf.reshape(h, [-1, h_dim])#  [b*v, h]

        with tf.variable_scope("gru_scope") as scope:
            for i in range(self.params['num_timesteps']):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                for edge_type in range(self.num_edge_types):
                    m = tf.matmul(h, tf.nn.dropout(self.weights['edge_weights'][edge_type],keep_prob=self.placeholders['dropout_keep_prob']))
                    m = tf.reshape(m, [-1, v, h_dim])
                    if self.params['use_edge_bias']:
                        m += self.weights['edge_biases'][edge_type]
                    if edge_type == 0:
                        acts = tf.matmul(self.__adjacency_matrix[edge_type], m)
                    else:
                        acts += tf.matmul(self.__adjacency_matrix[edge_type], m)
                acts = tf.reshape(acts, [-1, h_dim])

                h = self.weights['node_gru'](acts, h)[1]
            last_h = tf.reshape(h, [-1, v, h_dim])
        return last_h



    def convertTargetToInputDec(self,target):
        after_slice = tf.strided_slice(target, [0, 0], [self.params["win"], -1], [1, 1])
        after_concat = tf.concat([tf.fill([self.params["win"], 1], 0.0), after_slice], 1)
        return after_concat

    def compressGr(self,last_h):
        last_h=tf.reduce_mean(last_h,axis=1)  # b,h
        return last_h




    def process_raw_Graphs(self, raw_data: Sequence[Any], is_training_data: bool, bucket_sizes=None) -> Any:
        bucketed={}
        for i in range(len(raw_data)):
            d=raw_data[i]
            adjMat=graph_to_adj_mat(d['graph'], self.max_num_vertices, self.num_edge_types, self.params['tie_fwd_bkwd'])
            bucketed[i]={
                'adj_mat': adjMat,
                'targets':d["targets"]# class label
            }



        return bucketed



    def pad_annotations(self, annotations):
        for node in annotations:
            length=len(node)
            for __ in range(self.params['hidden_size']-length):
                node.append(0)
        annotations=np.asarray(annotations,dtype=np.float32)
        return annotations


    def make_batch(self, elements,end_data,elements__target):
        batch_data = {'adj_mat': [], 'labels': [],'dec_input':[]}
        last_seq_index_in_batch = len(elements__target) - 1
        first = np.zeros(shape=(self.max_num_vertices * self.max_num_vertices), dtype=np.float32)
        seq_ind=0
        for seq,seq_target in zip(elements,elements__target):
            batch_data['dec_input'].append(first)
            for ind_dy_graph,d in enumerate(seq):
                batch_data['adj_mat'].append(d['adj_mat'])
                if(ind_dy_graph<len(seq_target)):
                    tt=decoder_output(seq_target[ind_dy_graph]['adj_mat'])
                    batch_data['labels'].append(tt)
                if(ind_dy_graph<len(seq_target)-1):
                    batch_data['dec_input'].append(tt)
            seq_ind=seq_ind+1

        return batch_data

    def make_pad(self, batch_data,index,len_last_seq_in_batch,max_len_seq_in_batch):
        jj=0
        while(jj<index):
            amat = np.zeros((self.num_edge_types, self.max_num_vertices, self.max_num_vertices))
            target=[]
            for i in range(len(amat[0])):
                for j in range(len(amat[0][i])):
                    target.append(0)
            target = np.asarray(target, dtype=np.float32)
            if(max_len_seq_in_batch>len_last_seq_in_batch):
                batch_data['labels'].append(target)
                len_last_seq_in_batch=len_last_seq_in_batch+1
            if (jj < index ):
                batch_data['dec_input'].append(target)
            jj=jj+1
        return batch_data

    def make_minibatch_iterator(self, data, is_training: bool):
        dropout_keep_prob = self.params['dropout'] if is_training else 1.
        step=0
        end_target_idx=0
        while(step<len(data) and end_target_idx<len(data)):
            elements_batch=[]
            elements_batch_target=[]

            timeSteps_TargetBatch=[]
            num_graphs=0
            end_data=False
            for b_nu in range(self.params['batch_size']):
                start_idx = step
                end_idx = step + self.params["win"]
                elements = []
                elements_target = []

                for ind in range(start_idx, end_idx):
                    if (ind < len(data)):
                        elements.append(data[ind])


                start_target_idx=end_idx
                end_target_idx = start_target_idx + self.params["win"]
                for ind in range(start_target_idx, end_target_idx):
                    if (ind < len(data)):
                        elements_target.append(data[ind])
                    else:
                        index_target=end_target_idx-ind
                        len_last_seq_in_batch_target=ind-start_target_idx
                        end_data=True
                        break



                step=step+self.params["overlap"]
                elements_batch.append(elements)
                elements_batch_target.append(elements_target)
                timeSteps_TargetBatch.append(len(elements_target))
                if(end_data==True):
                    break
            if (end_data == False):
                index_target=self.params["win"]


            for seq in elements_batch_target:
                num_graphs=num_graphs+len(seq)

            batch_data = self.make_batch(elements_batch,end_data,elements_batch_target)

            if(end_data == True):
                batch_data=self.make_pad(batch_data,index_target,len_last_seq_in_batch_target,max(timeSteps_TargetBatch))
                num_graphs_init=num_graphs+index_target
            else:
                num_graphs_init=num_graphs


            initial_representations =[self.node_features[k] for k in range(self.max_num_vertices)]
            initial_representations = self.pad_annotations(initial_representations)
            initial_representations_final=[]
            for ind in range(num_graphs_init):
                initial_representations_final.append(initial_representations)

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: initial_representations_final,
                self.placeholders['target_values']: batch_data['labels'],
                self.placeholders['num_graphs']: num_graphs,
                self.placeholders['num_vertices']: self.max_num_vertices,
                self.placeholders['adjacency_matrix']: batch_data['adj_mat'],
                self.placeholders['dropout_keep_prob']: dropout_keep_prob,
                self.placeholders['target_sequence_length']:timeSteps_TargetBatch,
                self.placeholders['decoder_input']:batch_data['dec_input'],
                self.placeholders['ini_state']: np.zeros(shape=(1,self.params['hidden_size']),dtype=np.float32),
                self.placeholders['ini_att']: np.zeros(shape=(1,self.params['win']), dtype=np.float32)

            }
            yield batch_feed_dict

        



def main():
    parameters = OrderedDict()
    try:
        optparser = optparse.OptionParser()
        optparser.add_option("-w", "--win", default="50",type='int',help="Window size")
        optparser.add_option("-b", "--batch", default="10",type='int',help="Batch size")
        optparser.add_option("-s", "--hidden_size", default="100",type='int',help="lstm hidden size")
        optparser.add_option("-n", "--num_layers", default="1",type='int',help="Number of encoder-decoder layers")
        optparser.add_option("-l", "--lr", default="0.001",type='float',help="Learning rate")
        optparser.add_option("-e", "--epoch", default="1",type='int',help="Number of epochs")
        optparser.add_option("-t", "--dir_training", default="../train",type='string',help="Training data directory")
        optparser.add_option("-v", "--dir_test", default="../test",type='string',help="Test data directory")
        optparser.add_option("-p", "--helper", default="th",help="Training helper")
        optparser.add_option("-d", "--dropout", default="0.5",type='float', help="Keep prob dropout, if 1 no dropout")
        optparser.add_option("-o", "--overlap", default="1",type='int', help="the overlap that we need to slid the window; it should be less than window size")


        opts = optparser.parse_args()[0]
        parameters['win'] = opts.win
        parameters['batch'] = opts.batch
        parameters['num_layers'] = opts.num_layers
        parameters['hidden_size'] = opts.hidden_size
        parameters['lr'] = opts.lr
        parameters['epoch'] = opts.epoch
        parameters['dir_training'] = opts.dir_training
        parameters['dir_test'] = opts.dir_test
        parameters['helper'] = opts.helper
        parameters['dropout']=opts.dropout
        parameters['overlap']=opts.overlap


        model = GGNNModel(parameters)
        model.train()  #
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
