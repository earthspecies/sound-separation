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
from sklearn import model_selection,cluster,pipeline,metrics,preprocessing,decomposition 
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
    nsources = 4
    model_dict = {4:'3223090',8:'2178900'}

    if os.path.isdir(conf["dataset_path"]):
        conf["subdataset"] = 'AL'
        conf["individual"] = 'Naranja'
        conf["input_files_path"] = os.path.join(conf["dataset_path"],'19',conf["subdataset"], conf["individual"])
        conf["annotation_path"] = os.path.join(conf["dataset_path"],'labelled_datasets','individual_vocalizations','classification_with_file.csv')
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

    filelist_train, filelist_test = sklearn.model_selection.train_test_split(filelist,
        random_state=42, 
        test_size=0.8, 
        shuffle=True)
    filelist_train = filelist

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
        #load annotations from csv
        #in the future replace with: https://github.com/BirdVox/PCEN-SNR
        df = pd.read_csv(conf["annotation_path"])
        #segmentation and grouping of short events 
        for sd,ft in filelist_train:
            individual = sd.split(os.path.sep)[-1]
            subdataset = sd.split(os.path.sep)[-2]
            df_file = df[df['file'].str.contains(ft)]
            df_file = df_file[df_file['file'].str.contains(individual)]
            df_file = df_file[df_file['file'].str.contains(subdataset)]
            df_file.reset_index(drop=True, inplace=True)

            #join events close apart in time
            index = 1
            while index < len(df_file):
                #print('{}->{}',format(df_file.iloc[index-1]['end.in.file']),df_file.at[index,'start.in.file'])
                if df_file.at[index,'start.in.file'] - df_file.iloc[index-1]['end.in.file'] < 10:
                    df_file.at[index-1,'end.in.file'] = df_file.at[index,'end.in.file']
                    df_file.drop(index, inplace=True)
                    df_file.reset_index(drop=True, inplace=True)
                    #print(len(df_file))
                else:
                    index += 1

            input_mix,sample_rate = librosa.load(os.path.join(sd,ft))
            ### DC offset removal
            freq_cut=10
            [b,a] = scipy.signal.butter(2, freq_cut/(sample_rate/2), 'highpass');
            input_mix = scipy.signal.filtfilt(b,a,input_mix)
                
            for i in range(len(df_file)):
                start = int((df_file.at[i,'start.in.file']-0.2) * sample_rate)
                end = int((df_file.at[i,'end.in.file']+0.5) * sample_rate)
                input_wav = input_mix[np.newaxis,np.newaxis,start:end]
                print('{}-{}-{}.wav'.format(ft[:-4],int(df_file.at[i,'start.in.file']),int(df_file.at[i,'end.in.file'])))
                #import pdb;pdb.set_trace()
                separated_waveforms = sess.run(
                    output_tensor,
                    feed_dict={input_placeholder: input_wav})[0]
                #sf.write(os.path.join(conf["save_dir"],'{}-{}-{}.wav'.format(ft[:-4],int(df_file.at[i,'start.in.file']),int(df_file.at[i,'end.in.file']))),np.squeeze(input_wav), sample_rate, 'PCM_24')
                # features = np.zeros((4, 4))
                # featuresc = np.zeros((4, 4))
                # for s in range(nsources):
                #     sf.write(os.path.join(conf["save_dir"],'{}-{}-{}-source{}.wav'.format(ft[:-4],int(df_file.at[i,'start.in.file']),int(df_file.at[i,'end.in.file']),s+1)), separated_waveforms[s], sample_rate, 'PCM_24')
                # #     features[s,0] = SNR(separated_waveforms[s],np.squeeze(input_wav))
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
                pca = sklearn.decomposition.PCA(n_components=nsources,random_state=42)
                # pca.fit(masks_r)
                # X = pca.transform(masks_r)
                km = sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=30, max_iter=100, random_state=42)
                pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.Normalizer(), pca, km)
                pipe.fit(masks_r)
                print(km.labels_)
                if km.labels_.sum() > 1:
                    audio_id = 0
                else:
                    audio_id =  [index for index,value in enumerate(km.labels_) if value == 1][0]
                clean = separated_waveforms[audio_id]
                noise = np.sum(separated_waveforms[[index for index in range(nsources) if index!=audio_id]],axis=0) 
                sf.write(os.path.join(conf["save_dir"],'{}-{}-{}-clean.wav'.format(ft[:-4],int(df_file.at[i,'start.in.file']),int(df_file.at[i,'end.in.file']))), clean, sample_rate, 'PCM_24')
                #sf.write(os.path.join(conf["save_dir"],'{}-{}-{}-noise.wav'.format(ft[:-4],int(df_file.at[i,'start.in.file']),int(df_file.at[i,'end.in.file']))), noise, sample_rate, 'PCM_24')
               
    

  


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    main(arg_dic)

#########
# python mixit_separation.py --dataset_path /home/marius/data/spanish-carrion-crows/ --model_dir /home/marius/data/bird_mixit_model_checkpoints/
