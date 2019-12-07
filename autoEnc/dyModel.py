#!/usr/bin/env/python

from typing import Tuple, List, Any, Sequence

import tensorflow as tf
import time
import os
import json
import numpy as np
import random
from tensorflow.python.layers.core import Dense
import logging
from basicDec import BasicDecoder
from decoder import dynamic_decode
from utils import ThreadedIterator


class Model(object):
    @classmethod
    def default_params(cls):
        return {
            'patience': 25,
            'clamp_gradient_norm': 1.0,
            'num_timesteps': 4,
            'use_graph': True,
            'tie_fwd_bkwd': True,
            'task_ids': [0],
            'random_seed': 0,
        }

    def __init__(self, parameters):
        params = self.default_params()
        params["train_dir"] =parameters['dir_training']
        params["val_dir"] = parameters['dir_test']
        params["learning_rate"]=parameters['lr']
        params["hidden_size"]=parameters['hidden_size']
        params["batch_size"]=parameters['batch']
        params["num_epochs"]=parameters["epoch"]
        params["win"]=parameters["win"]
        params["num_layers"]=parameters["num_layers"]
        params["helper"]=parameters['helper']
        params["dropout"]=parameters['dropout']
        params["overlap"]=parameters['overlap']

        self.run_id = "_".join(["Win",str(params["win"]),".NuLayers",str(params["num_layers"]),".LSTMhid",str(params["hidden_size"]),".Helper",str(params["helper"]),".Batch",str(params["batch_size"]),".Epoch",str(params["num_epochs"]),".LR",str(params["learning_rate"]),".Drop"+str(params["dropout"]),".Overlap",str(params["overlap"]),time.strftime("%Y-%m-%d-%H-%M-%S")])
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s',filename='log/'+self.run_id+".txt",filemode='w')
        self.params = params
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.train_data = self.load_Data(params["train_dir"], is_training_data=True)
        self.valid_data = self.load_Data(self.params["val_dir"], is_training_data=False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()
            self.initialize_model()

    def loadDataAccToGGNN(self,file):
        f=open(file,"r")
        lines=f.readlines()
        edges=[]
        nodes=[]
        for line in lines:
            line=line.split(" ")
            if(int(line[0]) not in nodes):
                nodes.append(int(line[0]))
            for elem in line[1:-1]:
                edges.append([int(line[0]),1,int(elem)])
                if (int(elem) not in nodes):
                    nodes.append(int(elem))
        return edges,nodes

    def load_Data(self, dir, is_training_data: bool):
        print("Loading data from %s" % dir)
        data={}
        #LOAD the data
        #data[folderNu][0][timestep][0]:list of nodes
        #data[folderNu][0][timestep][1]:list of edges
        for subj in os.listdir(dir):
            print(subj)
            subjNu=int(subj.split("-")[0])
            target=int(subj.split("-")[1])
            data[subjNu] =[{},target]
            for time in os.listdir(dir+subj):
                grNu=int(time)
                data[subjNu][0][grNu]=[]
                edges,nodes=self.loadDataAccToGGNN(dir+subj+"/"+time)
                data[subjNu][0][grNu].append(edges)
                data[subjNu][0][grNu].append(nodes)
        sortedGrs=[]
        classNu=4
        targetIndex=0
        if(is_training_data):
            self.targetDic={}
        for i in range(max(data.keys())+1):
            for j in range(max(data[i][0])+1):
                if (is_training_data):
                    if(data[i][1] not in self.targetDic.keys()):
                        vec = [0 for i in range(classNu)]
                        vec[targetIndex] = 1
                        self.targetDic[data[i][1]]=vec
                        targetIndex=targetIndex+1
                sortedGrs.append({"graph":data[i][0][j][0],"nodes":data[i][0][j][1],"targets":data[i][1]})


        if(is_training_data):
            verticesID = []
            self.node_features={}
            for g in sortedGrs:
                for e in g['graph']:
                    for v in [e[0],e[2]]:
                        if(v not in verticesID):
                            verticesID.append(v)
            self.max_num_vertices=len(verticesID)

            for vertex in verticesID:
                vec=[0 for i in range(self.max_num_vertices)]
                vec[vertex]=1
                self.node_features[vertex]=vec


            num_fwd_edge_types = 0
            for g in sortedGrs:
                num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
            self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
            self.annotation_size = len(self.node_features[0])
        if(is_training_data):
            self.whole_nu_grs_training=len(sortedGrs)
        else:
            self.whole_nu_grs_validation=len(sortedGrs)

        return self.process_raw_Graphs(sortedGrs, is_training_data)



    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def compressGr(self,last_h):
        raise Exception("Models have to implement whole graph embedding!")

    def computePredictionDist(self,grRep,weights):
        raise Exception("Models have to implement whole predicrtion distribution!")
    def convertTargetToInputDec(self):
        raise Exception("Models have to implement decoder input convertor!")

    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [ None,self.max_num_vertices*self.max_num_vertices],name='target_values')
        self.placeholders['decoder_input'] = tf.placeholder(tf.float32, [None,self.max_num_vertices*self.max_num_vertices],name='decoder_input')
        self.placeholders['ini_state']=tf.placeholder(tf.float32,[None,self.params['hidden_size']],name='ini_state')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')

        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['initial_node_representation'])

        self.ops['losses'] = []
        self.placeholders['target_sequence_length'] = tf.placeholder(tf.int32, [None], name='target_sequence_length')
        grRep=self.compressGr(self.ops['final_node_representations'])#--,h
        grRep=tf.reshape(grRep,[-1,self.params["win"],self.params['hidden_size']])# b,w,h
        self.placeholders['static_gr_rep']=grRep

        stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.params['hidden_size']), self.params["dropout_keep_prob"]) for _ in range(self. params["num_layers"])])
        outputs_enc, encoder_state = tf.nn.dynamic_rnn(stacked_cells,grRep,sequence_length=self.placeholders['target_sequence_length'] ,dtype=tf.float32)
        self.placeholders['enc_output']=outputs_enc
        dec_embed_input=tf.reshape(self.placeholders['decoder_input'],shape=(-1,self.params["win"],self.max_num_vertices*self.max_num_vertices))
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.params['hidden_size']) for _ in range(self. params["num_layers"])])
        dec_cell = tf.contrib.rnn.DropoutWrapper(cells, output_keep_prob=self.params["dropout_keep_prob"])
        max_target_len = tf.reduce_max(self.placeholders['target_sequence_length'])

        if(self.params["helper"]=="th"):
            helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, self.placeholders['target_sequence_length'])
        else:
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(dec_embed_input,self.placeholders['target_sequence_length'], sampling_probability=1.0)

        output_layer = Dense(self.max_num_vertices*self.max_num_vertices)
        decoder = BasicDecoder(dec_cell, helper, encoder_state,self.placeholders['ini_state'],output_layer)
        outputs_dec, state_Dec, _ ,full_States= dynamic_decode(decoder,max_target_len,self.params["hidden_size"], impute_finished=True,maximum_iterations=max_target_len)
        self.placeholders['dec_output']=outputs_dec
        self.placeholders['state_Dec']=full_States

        masks = tf.sequence_mask(self.placeholders['target_sequence_length'], max_target_len, dtype=tf.float32,name='masks')#b,tl(w)
        masks=tf.expand_dims(masks,2)
        training_logits = tf.identity(outputs_dec.rnn_output,name='logits')
        training_logits=training_logits*masks
        training_logits=tf.reshape(training_logits,(-1,self.max_num_vertices*self.max_num_vertices))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=training_logits,labels=self.placeholders['target_values'] )
        loss=tf.reduce_sum(loss,axis=1)
        loss=tf.reduce_mean(loss)


        self.ops['loss'] = loss

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        self.sess.run(tf.local_variables_initializer())
    def crossEntropy(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")


    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def finalRep(self,output,grReps,start_idx,step,lastSeenGr):
        #output:b,w,h;
        for seq in output:
            for gr in seq:
                if (len(grReps) < start_idx + 1):
                    grReps.append([])
                if(start_idx+1>lastSeenGr):
                    grReps[start_idx] = gr
                start_idx = start_idx + 1
            step = step + self.params["overlap"]
            lastSeenGr=start_idx
            start_idx = step
        return grReps,start_idx,step,lastSeenGr

    def writeGrRepSVMformat(self,grReps,data,name,type,training):
        fw = open("result/"+self.run_id + "." + name + "."+type+".txt", "w")
        if(training==True):
            whole_nu_grs=self.whole_nu_grs_training
        else:
            whole_nu_grs=self.whole_nu_grs_validation
        for i in range(whole_nu_grs):
            grRep = grReps[i]
            fw.write(str(data[i]["targets"]))
            for j in range(len(grRep)):
                fw.write(" " + str(j + 1) + ":" + str(grRep[j]))
            fw.write("\n")
        fw.flush()
        fw.close()

    def findRep(self,data,name,training):
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, False), max_queue_size=5)
        grReps_enc=[]
        grReps_dec=[]
        grReps_static=[]
        step_enc = 0
        step_dec = 0
        step_static = 0

        start_idx_enc = step_enc
        start_idx_dec = step_dec
        start_idx_static = step_static

        lastSeenGr_enc=-1
        lastSeenGr_dec=-1
        lastSeenGr_static=-1


        for batch_ind, batch_data in enumerate(batch_iterator):
            enc_output,static_grRep,_,state_dec = self.sess.run((self.placeholders['enc_output'],self.placeholders['static_gr_rep'],self.placeholders['dec_output'],self.placeholders['state_Dec']), feed_dict=batch_data)
            grReps_enc,start_idx_enc,step_enc,lastSeenGr_enc=self.finalRep(enc_output,grReps_enc,start_idx_enc,step_enc,lastSeenGr_enc)
            grReps_dec,start_idx_dec,step_dec,lastSeenGr_dec=self.finalRep(state_dec,grReps_dec,start_idx_dec,step_dec,lastSeenGr_dec)
            grReps_static,start_idx_static,step_static,lastSeenGr_static=self.finalRep(static_grRep,grReps_static,start_idx_static,step_static,lastSeenGr_static)

        self.writeGrRepSVMformat(grReps_enc,data,name,"enc",training)
        self.writeGrRepSVMformat(grReps_dec,data,name,"dec",training)
        self.writeGrRepSVMformat(grReps_static,data,name,"static",training)

        print(str(len(data)))



    def run_epoch(self, epoch_name: str, data):
        loss = 0
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, True), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            fetch_list = [self.ops['loss'], self.ops['train_step'],self.placeholders['state_Dec']]


            result = self.sess.run(fetch_list, feed_dict=batch_data)
            batch_loss = result[0]
            loss += batch_loss * num_graphs
            print("Running epoch: "+str(epoch_name)+", batch "+str(step)+", num_graphs: "+str(num_graphs)+", loss so far: "+str(loss/processed_graphs))


        loss = loss / processed_graphs
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return loss, instance_per_sec

    def train(self):
        total_time_start = time.time()
        with self.graph.as_default():
            (best_val_acc, best_val_acc_epoch) = (float("+inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_speed = self.run_epoch("epoch %i (training)" % epoch,self.train_data)
                print("\r\x1b[K Train: loss: %.5f | instances/sec: %.2f" % (train_loss,train_speed))
                logging.info("Training Epoch: "+str(epoch)+" ,Train Loss: "+str(train_loss)+" , Train Speed: "+str(train_speed))

                train_loss, train_speed = self.run_epoch("epoch %i (validation)" % epoch, self.valid_data)
                print("\r\x1b[K Validation: loss: %.5f | instances/sec: %.2f" % (train_loss, train_speed))
                epoch_time = time.time() - total_time_start
                logging.info("Validation Epoch: "+str(epoch)+" ,Train Loss: "+str(train_loss)+" , Train Speed: "+str(train_speed))

            nameOut=self.params["train_dir"].split("/")[-2]
            self.findRep(self.train_data,nameOut,True)
            nameOut=self.params["val_dir"].split("/")[-2]
            self.findRep(self.valid_data,nameOut,False)


    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

