#-------------------------------------
# This is unofficial implementation of 2018 CVPR paper: Learning to Compare: Relation Network for Few-Shot Learning
# For official implementation, please visit: https://github.com/floodsung/LearningToCompare_FSL.git
#-------------------------------------


import tensorflow as tf
from tensorflow.keras import layers
import task_generator_test as tg
import math
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import scipy as sp
import scipy.stats


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 10)
parser.add_argument("-e","--episode",type = int, default= 10)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
HIDDEN_UNIT = args.hidden_unit

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h


@tf.function
def test(feature_encoder, relation_network, samples, sample_labels, batches, batch_labels):
    sample_features = feature_encoder(samples,True)
    batch_features = feature_encoder(batches,True)

    sample_features_ext = tf.repeat(tf.expand_dims(sample_features,0),3*CLASS_NUM,axis=0)
    batch_features_ext = tf.repeat(tf.expand_dims(batch_features,0),CLASS_NUM,axis=0)
    batch_features_ext = tf.transpose(batch_features_ext,(1,0,2,3,4))
    relation_pairs = tf.reshape(tf.concat([sample_features_ext,batch_features_ext],axis=4),(-1,19,19,FEATURE_DIM*2))
    relations = tf.reshape(relation_network(relation_pairs,True),(-1,CLASS_NUM))

    labels = tf.math.argmax(tf.squeeze(batch_labels),1)
    predict_labels = tf.math.argmax(relations,1)

    rewards = [1.0 if predict_labels[i] == labels[i] else 0.0 for i in range(15)]

    return np.sum(rewards)/15.0


def main():
    # Step 1: init data folders
    print("init data folders")
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders()
    # init character folders for dataset construction

    # Step 2: init neural networks
    print("load neural networks")
    if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot")):
        feature_encoder = tf.keras.models.load_model(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot"))
        print("load feature encoder success")
    if os.path.exists(str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot")):
        relation_network = tf.keras.models.load_model(str("./models/miniimagenet_relation_network_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot"))
        print("load relation network success")

    # Step 3: build graph

    total_accuracy = 0.0
    for episode in range(EPISODE):
        print("Testing...")
        accuracies = []
        for i in range(TEST_EPISODE):      
            task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
            sample_dataset = tg.dataset(task,SAMPLE_NUM_PER_CLASS,split='train',shuffle=False)
            batch_dataset = tg.dataset(task,3,split='test',shuffle=True)

            sample_dataloader = tf.data.Dataset.from_generator(sample_dataset.generator, output_types=(tf.float32, tf.float32), output_shapes = ((84,84,3),(5,1))).batch(SAMPLE_NUM_PER_CLASS*CLASS_NUM).take(1)
            batch_dataloader = tf.data.Dataset.from_generator(batch_dataset.generator, output_types=(tf.float32, tf.float32), output_shapes = ((84,84,3),(5,1))).batch(3*CLASS_NUM).take(1)

            samples,sample_labels = next(iter(sample_dataloader))
            batches,batch_labels = next(iter(batch_dataloader)) 

            accuracies.append(test(feature_encoder, relation_network, samples, sample_labels, batches, batch_labels))
        
        test_accuracy,h = mean_confidence_interval(accuracies)

        print("test accuracy:",test_accuracy,"h:",h)

        total_accuracy += test_accuracy

    print("aver_accuracy:",total_accuracy/EPISODE)


if __name__ == '__main__':
    main()
