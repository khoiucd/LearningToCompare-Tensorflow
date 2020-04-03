#-------------------------------------
# This is unofficial implementation of 2018 CVPR paper: Learning to Compare: Relation Network for Few-Shot Learning
# For official implementation, please visit: https://github.com/floodsung/LearningToCompare_FSL.git
#-------------------------------------


import tensorflow as tf
from tensorflow.keras import layers
import task_generator as tg
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
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 10)
parser.add_argument("-e","--episode",type = int, default= 500000)
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


class CNNEncoder(tf.keras.Model):
    def __init__(self):
        super(CNNEncoder, self).__init__(name='CNNEncoder')

        self.conv2a = layers.Conv2D(filters=64,input_shape=(84,84,3),kernel_size=(3,3),padding='valid',bias_initializer='glorot_uniform')
        self.bn2a = layers.BatchNormalization(axis=3,momentum=0.0,epsilon=1e-05,fused=False)
        self.pool2a = layers.MaxPool2D(pool_size=(2,2))

        self.conv2b = layers.Conv2D(filters=64,kernel_size=(3,3),padding='valid',bias_initializer='glorot_uniform')
        self.bn2b = layers.BatchNormalization(axis=3,momentum=0.0,epsilon=1e-05,fused=False)
        self.pool2b = layers.MaxPool2D(pool_size=(2,2))

        self.conv2c = layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',bias_initializer='glorot_uniform')
        self.bn2c = layers.BatchNormalization(axis=3,momentum=0.0,epsilon=1e-05,fused=False)

        self.conv2d = layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',bias_initializer='glorot_uniform')
        self.bn2d = layers.BatchNormalization(axis=3,momentum=0.0,epsilon=1e-05,fused=False)

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x,training)
        x = tf.nn.relu(x)
        x = self.pool2a(x)

        x = self.conv2b(x)
        x = self.bn2b(x,training)
        x = tf.nn.relu(x)
        x = self.pool2b(x)

        x = self.conv2c(x)
        x = self.bn2c(x,training)
        x = tf.nn.relu(x)

        x = self.conv2d(x)
        x = self.bn2d(x,training)

        return tf.nn.relu(x)


class RelationNetwork(tf.keras.Model):
    def __init__(self):
        super(RelationNetwork, self).__init__(name='RelationNetwork')

        self.conv2a = layers.Conv2D(filters=64,input_shape=(19,19,128),kernel_size=(3,3),padding='valid',bias_initializer='glorot_uniform')
        self.bn2a = layers.BatchNormalization(axis=3,momentum=0.0,epsilon=1e-05,fused=False)
        self.pool2a = layers.MaxPool2D(pool_size=(2,2))

        self.conv2b = layers.Conv2D(filters=64,kernel_size=(3,3),padding='valid',bias_initializer='glorot_uniform')
        self.bn2b = layers.BatchNormalization(axis=3,momentum=0.0,epsilon=1e-05,fused=False)
        self.pool2b = layers.MaxPool2D(pool_size=(2,2))

        self.fc1 = layers.Dense(8,activation='relu')
        self.fc2 = layers.Dense(1,activation='sigmoid') 

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x,training)
        x = tf.nn.relu(x)
        x = self.pool2a(x)

        x = self.conv2b(x)
        x = self.bn2b(x,training)
        x = tf.nn.relu(x)
        x = self.pool2b(x)

        x = layers.Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


@tf.function
def train_one_step(feature_encoder, relation_network, feature_encoder_optim, relation_network_optim, samples, sample_labels, batches, batch_labels):
    with tf.GradientTape() as feature_encoder_tape, tf.GradientTape() as relation_network_tape:
        sample_features = feature_encoder(samples,True)
        sample_features = tf.reshape(sample_features,(CLASS_NUM,SAMPLE_NUM_PER_CLASS,19,19,FEATURE_DIM))
        sample_features = tf.math.reduce_sum(sample_features, 1)
        batch_features = feature_encoder(batches,True)

        sample_features_ext = tf.repeat(tf.expand_dims(sample_features,0),BATCH_NUM_PER_CLASS*CLASS_NUM,axis=0)
        batch_features_ext = tf.repeat(tf.expand_dims(batch_features,0),CLASS_NUM,axis=0)

        batch_features_ext = tf.transpose(batch_features_ext,(1,0,2,3,4))
        relation_pairs = tf.reshape(tf.concat([sample_features_ext,batch_features_ext],axis=4),(-1,19,19,FEATURE_DIM*2))

        relations = tf.reshape(relation_network(relation_pairs,True),(-1,CLASS_NUM))

        mse = tf.keras.losses.MeanSquaredError()
        one_hot_labels = tf.squeeze(batch_labels)
        loss = mse(relations,one_hot_labels)

    grads_feature_encoder = feature_encoder_tape.gradient(loss, feature_encoder.trainable_variables)
    grads_relation_network = relation_network_tape.gradient(loss, relation_network.trainable_variables)

    feature_encoder_optim.apply_gradients(zip(grads_feature_encoder, feature_encoder.trainable_variables))
    relation_network_optim.apply_gradients(zip(grads_relation_network, relation_network.trainable_variables))
    return loss


@tf.function
def test(feature_encoder, relation_network, samples, sample_labels, batches, batch_labels):
    sample_features = feature_encoder(samples,True)
    sample_features = tf.reshape(sample_features,(CLASS_NUM,SAMPLE_NUM_PER_CLASS,19,19,FEATURE_DIM))
    sample_features = tf.math.reduce_sum(sample_features, 1)
    batch_features = feature_encoder(batches,True)

    sample_features_ext = tf.repeat(tf.expand_dims(sample_features,0),5*CLASS_NUM,axis=0)
    batch_features_ext = tf.repeat(tf.expand_dims(batch_features,0),CLASS_NUM,axis=0)
    batch_features_ext = tf.transpose(batch_features_ext,(1,0,2,3,4))
    relation_pairs = tf.reshape(tf.concat([sample_features_ext,batch_features_ext],axis=4),(-1,19,19,FEATURE_DIM*2))
    relations = tf.reshape(relation_network(relation_pairs,True),(-1,CLASS_NUM))

    labels = tf.math.argmax(tf.squeeze(batch_labels),1)
    predict_labels = tf.math.argmax(relations,1)

    rewards = [1.0 if predict_labels[i] == labels[i] else 0.0 for i in range(25)]

    return np.sum(rewards)/25.0


def main():
    # Step 1: init data folders
    print("init data folders")
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders()
    # init character folders for dataset construction

    # Step 2: init neural networks
    print("init neural networks")
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()

    feature_encoder_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(LEARNING_RATE,100000,0.5,staircase=True)
    feature_encoder_optim = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-08)
    relation_network_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(LEARNING_RATE,100000,0.5,staircase=True)
    relation_network_optim = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-08)

    if os.path.exists(str("models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot")):
        feature_encoder = tf.keras.models.load_model(str("models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot"))
        print("load feature encoder success")
    if os.path.exists(str("models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot")):
        relation_network = tf.keras.models.load_model(str("models/miniimagenet_relation_network_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot"))
        print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataset = tg.dataset(task,SAMPLE_NUM_PER_CLASS,split='train',shuffle=False)
        batch_dataset = tg.dataset(task,BATCH_NUM_PER_CLASS,split='test',shuffle=True)

        sample_dataloader = tf.data.Dataset.from_generator(sample_dataset.generator, output_types=(tf.float32, tf.float32), output_shapes = ((84,84,3),(5,1))).batch(SAMPLE_NUM_PER_CLASS*CLASS_NUM).take(1)
        batch_dataloader = tf.data.Dataset.from_generator(batch_dataset.generator, output_types=(tf.float32, tf.float32), output_shapes = ((84,84,3),(5,1))).batch(BATCH_NUM_PER_CLASS*CLASS_NUM).take(1)

        samples,sample_labels = next(iter(sample_dataloader))
        batches,batch_labels = next(iter(batch_dataloader))

        loss = train_one_step(feature_encoder, relation_network, feature_encoder_optim, relation_network_optim, samples, sample_labels, batches, batch_labels).numpy()

        if (episode+1)%100 == 0:
            print("episode:",episode+1,"loss",loss)

        if episode%5000 == 0:
            # test
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):      
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
                sample_dataset = tg.dataset(task,SAMPLE_NUM_PER_CLASS,split='train',shuffle=False)
                batch_dataset = tg.dataset(task,5,split='test',shuffle=True)

                sample_dataloader = tf.data.Dataset.from_generator(sample_dataset.generator, output_types=(tf.float32, tf.float32), output_shapes = ((84,84,3),(5,1))).batch(SAMPLE_NUM_PER_CLASS*CLASS_NUM).take(1)
                batch_dataloader = tf.data.Dataset.from_generator(batch_dataset.generator, output_types=(tf.float32, tf.float32), output_shapes = ((84,84,3),(5,1))).batch(5*CLASS_NUM).take(1)

                samples,sample_labels = next(iter(sample_dataloader))
                batches,batch_labels = next(iter(batch_dataloader)) 

                accuracies.append(test(feature_encoder, relation_network, samples, sample_labels, batches, batch_labels))
            
            test_accuracy,h = mean_confidence_interval(accuracies)
            
            print("test accuracy:",test_accuracy,"h:",h)

            if test_accuracy > last_accuracy:

                # save networks
                feature_encoder.save(str("models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot"),save_format='tf')
                relation_network.save(str("models/miniimagenet_relation_network_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot"),save_format='tf')

                print("save networks for episode:",episode)

                last_accuracy = test_accuracy


if __name__ == '__main__':
    main()
