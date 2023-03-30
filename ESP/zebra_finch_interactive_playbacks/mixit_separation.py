'''
python mixit_separation.py --dataset_path /home/jupyter/data/zebra_finch/Audio --model_dir /home/jupyter/data/bird_mixit_model_checkpoints/
'''
import os, sys
import argparse

import pandas as pd
import numpy as np
import scipy
import librosa
import csv
import soundfile as sf
from io import StringIO

import pcen_snr

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn import cluster, pipeline

import tensorflow as tf2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()

import tensorflow_hub


from contextlib import redirect_stdout

MIN_FILE_SIZE = 1000
ONSET_THRESHOLD = 4e-4
NSOURCES = 4

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path to zebra finch Audio folder containing Left and Right sub-folders"
)

parser.add_argument(
    "--model_dir", type=str, required=True, help="Local path to mixit model directory"
)


# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])

    return class_names

@tf.function
def compute(model, audio):
  scores, _, _ = model(audio)
  return scores

@tf.function
def main(conf):
   
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
    meta_graph_filename = os.path.join(conf["model_dir"], 'output_sources'+str(NSOURCES), 'inference.meta')
    tf.logging.info('Importing meta graph: %s', meta_graph_filename)
    with graph.as_default() as g:
        saver = tf.train.import_meta_graph(meta_graph_filename)
        meta_graph_def = g.as_graph_def()
    
    graph1 = tf.Graph()
    sess1 = tf.compat.v1.Session(graph=graph1)
    yamnet_model = tensorflow_hub.load('https://tfhub.dev/google/yamnet/1')
    # class_map_path = yamnet_model.class_map_path().numpy()
    # class_names = class_names_from_csv(class_map_path)

    #tf.compat.v1.disable_eager_execution()
    with sess as ss, sess1 as ss1:
        saver.restore(ss, os.path.join(conf["model_dir"],"output_sources"+str(NSOURCES),"model.ckpt-"+model_dict[NSOURCES]))
    
        input_tensor_name = 'input_audio/receiver_audio:0'
        output_tensor_name = 'denoised_waveforms:0'
        input_placeholder = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)
        mask_tensor = graph.get_tensor_by_name('Reshape_2:0') #spectrograms
        #mask_tensor = graph.get_tensor_by_name('mask:0') #masks 
        #get network parameters
        # import pdb;pdb.set_trace()
        # pars = [n.name for n in tf.get_default_graph().as_graph_def().node if hasattr(n, 'op') and n.op == 'Const']
        

        # md.sort_values(by='time',inplace=True)
        drop_first = True
        for index, row in md.iterrows():
            # if drop_first:
            #     md.drop([index],inplace=True) #drop first 30 seconds because it's an outlier - recording setup
            #     drop_first = False
            #     continue
            timestamp = row['timestamp']
            onsets = []
            audio = []
            mini_dict = {'Left':row['left_path'],'Right':row['right_path']}
            for c,p in mini_dict.items():
                input_mix,sample_rate = librosa.load(os.path.join(conf["dataset_path"],c,p))
                # ### DC offset removal
                lowcut=200 #Hz - higher frequency cutoff since these are bird sounds
                highcut=10000 #Hz - higher frequency cutoff since these are bird sounds
                [b,a] = scipy.signal.butter(4,[lowcut, highcut], fs=sample_rate, btype='band')
                input_wav = scipy.signal.lfilter(b,a,input_mix)
                
                # # rms is not a good measure because some signals have low amplitude while air conditioning ones have high amplitude
                # rms = librosa.feature.rms(S=librosa.magphase(librosa.stft(input_wav, window=np.ones, center=False))[0])
                # # zero crossing rate is not very stable, noise can have high zcr
                # zcr = librosa.feature.zero_crossing_rate(y=input_wav)
                # # noise may have low spectral flatness but so do some signals with sparse calls
                # sp = librosa.feature.spectral_flatness(S=librosa.feature.melspectrogram(y=input_wav, sr=sample_rate, n_mels=128, fmin=1000, fmax=10000, hop_length=512))
                # spectral flux onset seems to be the best feature but the threshold might be problematic with the sparse calls
                on = librosa.onset.onset_strength(S=librosa.feature.melspectrogram(y=input_wav, sr=sample_rate, n_mels=128, fmin=1000, fmax=10000, hop_length=512))
                onsets.append(np.mean(on))
                audio.append(input_wav)
            if onsets[0]<ONSET_THRESHOLD and onsets[1]<ONSET_THRESHOLD:
                print("Skipping "+timestamp+" because both channels have low onset strength.")
                continue


                # ### save audio
                # sf.write(os.path.join(conf["save_dir"],'{}-{}.wav'.format(timestamp,p.split('_',1)[0])),np.squeeze(input_wav), sample_rate, 'PCM_24')
                # for s in range(NSOURCES):
                #     sf.write(os.path.join(conf["save_dir"],'{}-{}-source{}.wav'.format(timestamp,p.split('_',1)[0],s+1)), separated_waveforms[s], sample_rate, 'PCM_24')
              
           
            centroids = 'k-means++'
            for i,c in enumerate(mini_dict):
                p = mini_dict[c]
                #import pdb;pdb.set_trace()
                input_wav = audio[i][np.newaxis,np.newaxis,:]
                # masks = sess.run(
                #     mask_tensor,
                #     feed_dict={input_placeholder: input_wav})[0]
                # masks_r = masks.reshape((NSOURCES, -1))
                # pca = sklearn.decomposition.PCA(n_components=NSOURCES,random_state=42)
                # km = sklearn.cluster.KMeans(n_clusters=2, init=centroids, n_init=30, max_iter=100, random_state=42)
                # pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.Normalizer(), pca, km)
                # pipe.fit(masks_r)
                # print(km.labels_)
                # #centroids  = km.cluster_centers_ 
                # if km.labels_.sum() == 1:
                #     audio_id =  [index for index,value in enumerate(km.labels_) if value == 1][0]  
                # elif km.labels_.sum() == NSOURCES-1:
                #     audio_id =  [index for index,value in enumerate(1-km.labels_) if value == 1][0]
                # else:
                #     import pdb;pdb.set_trace()  
                ### all together now
                separated_waveforms = sess.run(
                    output_tensor,
                    feed_dict={input_placeholder: input_wav})[0]
                
                for waveform in separated_waveforms:
                    waveform = waveform / tf.int16.max
                    
                    import pdb;pdb.set_trace()

                    outs = ss1.run(compute(yamnet_model,waveform), 
                        feed_dict={input_placeholder: waveform})
                    scores, _, _ = yamnet_model(waveform)
                    scores_np = scores.numpy()
                    infered_class = class_names[scores_np.mean(axis=0).argmax()]
                    print(f'The main sound is: {infered_class}')



                
                # clean = separated_waveforms[audio_id]
                # noise = np.sum(separated_waveforms[[index for index in range(NSOURCES) if index!=audio_id]],axis=0) 
                # sf.write(os.path.join(conf["save_dir"],'{}-{}.wav'.format(timestamp,p.split('_',1)[0])),np.squeeze(input_wav), sample_rate, 'PCM_24')
                # sf.write(os.path.join(conf["save_dir"],'{}-{}-clean.wav'.format(timestamp,p.split('_',1)[0])),np.squeeze(clean), sample_rate, 'PCM_24')    
                # sf.write(os.path.join(conf["save_dir"],'{}-{}-noise.wav'.format(timestamp,p.split('_',1)[0])),np.squeeze(noise), sample_rate, 'PCM_24')            
               

  


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    main(arg_dic)

#########
# python mixit_separation_vanilla.py --dataset_path /home/marius/data/spanish-carrion-crows/ --model_dir /home/marius/data/bird_mixit_model_checkpoints/
