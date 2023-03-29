'''
python mixit_separation.py --dataset_path /home/jupyter/data/zebra_finch/Audio --model_dir /home/jupyter/data/bird_mixit_model_checkpoints/
'''
import os, sys
import argparse

import pandas as pd
import numpy as np
import scipy
import librosa
import soundfile as sf
from io import StringIO

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn import cluster

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from contextlib import redirect_stdout

MIN_FILE_SIZE = 1000

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path to zebra finch Audio folder containing Left and Right sub-folders"
)

parser.add_argument(
    "--model_dir", type=str, required=True, help="Local path to mixit model directory"
)

def SNR(signal, noise):
    signal_mean=np.mean(10**signal)
    noise_mean=np.mean(10**noise)
    SNR = 10*np.log10(abs(signal_mean-noise_mean)/(noise_mean+1e-10))
    #print(str(round(SNR,2))+' '+'dB')
    return SNR

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def main(conf):
    nsources = 4
    model_dict = {4:'3223090',8:'2178900'}

    if os.path.isdir(conf["dataset_path"]):
        conf["save_dir"] = os.path.join(conf["dataset_path"],'separation')
        os.makedirs(conf["save_dir"], exist_ok=True)
    else:
        print("dataset_path is not valid.\n")
        sys.exit(1)

    #build dataset metadata
    left_path = os.path.join(conf["dataset_path"],'Left')
    left = {file.name.split('_',1)[1]:file.name.split('_',1)[0] for file in os.scandir(left_path) if file.name[-4:]=='.wav' and file.stat().st_size>MIN_FILE_SIZE and not file.name.startswith('.') and not file.is_dir()}
    right_path = os.path.join(conf["dataset_path"],'Right')
    right = {file.name.split('_',1)[1]:file.name.split('_',1)[0] for file in os.scandir(right_path) if file.name[-4:]=='.wav' and file.stat().st_size>MIN_FILE_SIZE and not file.name.startswith('.') and not file.is_dir()}
    md = pd.DataFrame(columns=['ID_left', 'ID_right', 'left_path', 'right_path','timestamp','time'])
    i=0
    for timestamp,id in (left.items()):
        if timestamp in right.keys():
            md.loc[i] = [id, right[timestamp], id+'_'+timestamp, right[timestamp]+'_'+timestamp, timestamp[:-4], pd.to_datetime(timestamp[:-4],format="%B_%d_%Y_%f")]
            i+=1
    
    #load tf mixit model
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    meta_graph_filename = os.path.join(conf["model_dir"], 'output_sources'+str(nsources), 'inference.meta')
    tf.logging.info('Importing meta graph: %s', meta_graph_filename)
    with graph.as_default() as g:
        saver = tf.train.import_meta_graph(meta_graph_filename)
        meta_graph_def = g.as_graph_def()

    with sess as ss:
        saver.restore(ss, os.path.join(conf["model_dir"],"output_sources"+str(nsources),"model.ckpt-"+model_dict[nsources]))
    
        input_tensor_name = 'input_audio/receiver_audio:0'
        output_tensor_name = 'denoised_waveforms:0'
        input_placeholder = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)
        mask_tensor = graph.get_tensor_by_name('Reshape_2:0') #spectrograms
        #mask_tensor = graph.get_tensor_by_name('mask:0') #masks 
        #get network parameters
        #[n.name for n in tf.get_default_graph().as_graph_def().node]

        #segmentation and grouping of short events 
        for index, row in md.iterrows():
            timestamp = row['timestamp']
            for c,p in {'Left':row['left_path'],'Right':row['right_path']}.items():
                input_mix,sample_rate = librosa.load(os.path.join(conf["dataset_path"],c,p))
                # ### DC offset removal
                # freq_cut=150 #Hz - higher frequency cutoff since these are bird sounds
                # [b,a] = scipy.signal.butter(2, freq_cut/(sample_rate/2), 'highpass')
                # input_wav = scipy.signal.filtfilt(b,a,input_mix)
                #import pdb;pdb.set_trace()
                ### feed forward audio through network
                input_wav = input_mix[np.newaxis,np.newaxis,:]
                separated_waveforms = sess.run(
                    output_tensor,
                    feed_dict={input_placeholder: input_wav})[0]
                sf.write(os.path.join(conf["save_dir"],'{}-{}.wav'.format(timestamp,p.split('_',1)[0])),np.squeeze(input_wav), sample_rate, 'PCM_24')
                # features = np.zeros((4, 4))
                # featuresc = np.zeros((4, 4))
                for s in range(nsources):
                    sf.write(os.path.join(conf["save_dir"],'{}-{}-source{}.wav'.format(timestamp,p.split('_',1)[0],s+1)), separated_waveforms[s], sample_rate, 'PCM_24')
                #     features[s,0] = SNR(separated_waveforms[s],np.squeeze(input_wav))
                #     featuresc[s,0] = np.correlate(separated_waveforms[s],np.squeeze(input_wav),'full').argmax()/len(separated_waveforms[s])
                #     features[s,1:4] = np.array([SNR(separated_waveforms[s],separated_waveforms[s1]) for s1 in range(4) if s1 != s])
                #     featuresc[s,1:4] = np.array([np.correlate(separated_waveforms[s],separated_waveforms[s1], 'full').argmax()/len(separated_waveforms[s]) for s1 in range(4) if s1 != s])
                # km = sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=30, max_iter=100, random_state=42)
                # km.fit(features)
                # print(km.labels_)
                # km = sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=30, max_iter=100, random_state=42)
                # km.fit(featuresc)
                # print(km.labels_)
                # masks = sess.run(
                #     mask_tensor,
                #     feed_dict={input_placeholder: input_wav})[0]
                # masks_r = masks.reshape((nsources, -1))
                # # ica = sklearn.decomposition.FastICA(n_components=masks.shape[1],random_state=42,whiten='unit-variance')
                # # ica.fit(masks_r)
                # # X = ica.transform(masks_r)
                # # pca = sklearn.decomposition.SparsePCA(n_components=100,random_state=42)
                # # pca.fit(masks_r)
                # # X = pca.transform(masks_r)
                # km = sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=30, max_iter=100, random_state=42)
                # km.fit(masks_r)
                # print(km.labels_)
                # if km.labels_.sum() == 2:
                #     import pdb;pdb.set_trace()

  


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    main(arg_dic)

#########
# python mixit_separation_vanilla.py --dataset_path /home/marius/data/spanish-carrion-crows/ --model_dir /home/marius/data/bird_mixit_model_checkpoints/
