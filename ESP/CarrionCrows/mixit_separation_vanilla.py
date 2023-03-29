import os, sys
import argparse
import tqdm
import pandas as pd
import numpy as np
import scipy
from scipy import signal
import librosa
import soundfile as sf
from io import StringIO
import seaborn as sns
import collections
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection 
from sklearn import cluster 
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from contextlib import redirect_stdout


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path to spanish-carrion-crow dataset"
)

parser.add_argument(
    "--model_dir", type=str, required=True, help="Local path to mixit model directory"
)

def autocorrelation(y, sr):
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                        hop_length=hop_length)
    # Compute global onset autocorrelation
    ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    ac_global = librosa.util.normalize(ac_global)
    tempo = librosa.feature.tempo(onset_envelope=oenv, sr=sr,
                              hop_length=hop_length)[0]
    fig, ax = plt.subplots(nrows=4, figsize=(10, 10))

    times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)

    ax[0].plot(times, oenv, label='Onset strength')

    ax[0].label_outer()

    ax[0].legend(frameon=True)

    librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,

                            x_axis='time', y_axis='tempo', cmap='magma',

                            ax=ax[1])

    ax[1].axhline(tempo, color='w', linestyle='--', alpha=1,

                label='Estimated tempo={:g}'.format(tempo))

    ax[1].legend(loc='upper right')

    ax[1].set(title='Tempogram')

    x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,

                    num=tempogram.shape[0])

    ax[2].plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')

    ax[2].plot(ac_global, '--', alpha=0.75, label='Global autocorrelation')

    ax[2].set(xlabel='Lag (seconds)')

    ax[2].legend(frameon=True)

    freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)

    ax[3].semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),

                label='Mean local autocorrelation', base=2)

    ax[3].semilogx(ac_global[1:], '--', alpha=0.75,

                label='Global autocorrelation', base=2)

    ax[3].axvline(tempo, color='black', linestyle='--', alpha=.8,

                label='Estimated tempo={:g}'.format(tempo))

    ax[3].legend(frameon=True)

    ax[3].set(xlabel='BPM')

    ax[3].grid(True)
    plt.show()
    return tempo

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
    nsources = 8
    model_dict = {4:'3223090',8:'2178900'}

    if os.path.isdir(conf["dataset_path"]):
        conf["subdataset"] = 'AL'
        conf["individual"] = 'Test'
        conf["input_files_path"] = os.path.join(conf["dataset_path"],'19',conf["subdataset"], conf["individual"])
        conf["save_dir"] = os.path.join(conf["dataset_path"],'separation','19',conf["subdataset"], conf["individual"])
        os.makedirs(conf["save_dir"], exist_ok=True)
    else:
        print("dataset_path is not valid.\n")
        sys.exit(1)

    filelist = []
    for subdir, dirs, files in os.walk(conf["input_files_path"]):
        fileshere = [
            (subdir, filename) for filename in files if filename.endswith(".wav")
        ]
        filelist.extend(fileshere)

    if len(filelist) <5:
        filelist_train = filelist
    else:
        filelist_train, filelist_test = sklearn.model_selection.train_test_split(filelist,
            random_state=42, 
            test_size=0.2, 
            shuffle=True)

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
        for sd,ft in filelist_train:
            individual = sd.split(os.path.sep)[-1]
            subdataset = sd.split(os.path.sep)[-2]

            input_mix,sample_rate = librosa.load(os.path.join(sd,ft))
            ### DC offset removal
            freq_cut=10
            [b,a] = scipy.signal.butter(2, freq_cut/(sample_rate/2), 'highpass');
            input_wav = scipy.signal.filtfilt(b,a,input_mix)
            input_wav = input_mix[np.newaxis,np.newaxis,:]
            separated_waveforms = sess.run(
                output_tensor,
                feed_dict={input_placeholder: input_wav})[0]
            sf.write(os.path.join(conf["save_dir"],'{}.wav'.format(ft[:-4])),np.squeeze(input_wav), sample_rate, 'PCM_24')
            # features = np.zeros((4, 4))
            # featuresc = np.zeros((4, 4))
            for s in range(nsources):
                sf.write(os.path.join(conf["save_dir"],'{}-source{}.wav'.format(ft[:-4],s+1)), separated_waveforms[s], sample_rate, 'PCM_24')
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
            masks = sess.run(
                mask_tensor,
                feed_dict={input_placeholder: input_wav})[0]
            masks_r = masks.reshape((nsources, -1))
            # ica = sklearn.decomposition.FastICA(n_components=masks.shape[1],random_state=42,whiten='unit-variance')
            # ica.fit(masks_r)
            # X = ica.transform(masks_r)
            # pca = sklearn.decomposition.SparsePCA(n_components=100,random_state=42)
            # pca.fit(masks_r)
            # X = pca.transform(masks_r)
            km = sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=30, max_iter=100, random_state=42)
            km.fit(masks_r)
            print(km.labels_)
            if km.labels_.sum() == 2:
                import pdb;pdb.set_trace()

  


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    main(arg_dic)

#########
# python mixit_separation_vanilla.py --dataset_path /home/marius/data/spanish-carrion-crows/ --model_dir /home/marius/data/bird_mixit_model_checkpoints/
