'''
python filter_tracks.py --dataset_path /home/marius/data/zebra_finch/Pair3
assumes the separation has been done and the files are in the separation folder
'''
import os, sys
import argparse

import pandas as pd
import numpy as np
import scipy
import librosa
import csv
import soundfile as sf
import norbert
import csv

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn import cluster, pipeline


import tensorflow as tf
import tensorflow_hub as hub


#import vggish_slim, vggish_params

from contextlib import redirect_stdout

NSOURCES = 4
CLASSES = {97:"Turkey",98:"Gobble",99:"Duck",100:"Quack",101:"Goose",106:"Bird",107:"Bird vocalization, bird call, bird song",108:"Chirp, tweet",109:"Squawk",111:"Coo",112:"Crow",113:"Caw",127:"Frog"}
WIENER = False
THRESHOLD_MAX = 0.1
THRESHOLD_MEAN = 0.001

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path to zebra finch Audio folder containing Left and Right sub-folders"
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

def wiener_filter(pred_bird,pred_noise):
    mixture = pred_bird + pred_noise

    # Compute stft
    bird_spec = librosa.stft(pred_bird)
    noise_spec = librosa.stft(pred_noise)

    # Separate mags and phases
    bird_mag = np.abs(bird_spec)
    bird_phase = np.angle(bird_spec)
    noise_mag = np.abs(noise_spec)
    noise_phase = np.angle(noise_spec)

    # Preparing inputs for wiener filtering
    mix_spec = librosa.stft(mixture)
    #mix: (nb_frames, nb_bins, nb_channels)
    mix_spec = np.transpose(mix_spec, [2, 1, 0])
    sources = np.vstack([bird_mag[None], noise_mag[None]])
    #sources: shape=(nb_frames, nb_bins, nb_channels, nb_sources)
    sources = np.transpose(sources, [3, 2, 1, 0])

    # Wiener
    specs = norbert.wiener(sources, mix_spec, use_softmask=False, iterations=30)
    specs = np.transpose(sources, [3, 2, 1, 0])

    # Building output specs with filtered mags and original phases
    # import pdb;pdb.set_trace()
    bird_spec = np.abs(specs[0, :, :, :]) * np.exp(1j * bird_phase)
    noise_spec = np.abs(specs[1, :, :, :]) * np.exp(1j * noise_phase)
    pred_bird = librosa.istft(bird_spec)
    pred_noise = librosa.istft(noise_spec)

    return pred_bird, pred_noise

def threshold_activity(x, Tp, Ta):
    locs = scipy.signal.find_peaks(x,height = Tp)[0]
    if len(locs) > 1:
        y = (x > Ta) * 1
        act = np.diff(y)
        u = np.where(act == 1)[0]
        d = np.where(act == -1)[0]
        signal_length = len(x)
        if len(u) > 0 and len(d) > 0:
            if d[0] < u[0]:
                u = np.insert(u, 0, 0)
                
            if d[-1] < u[-1]:
                d = np.append(d, signal_length-1)
                
            starts = []
            ends = []
            
            activity = np.zeros(signal_length,)
            
            for candidate_up, candidate_down in zip(u, d):
                candidate_segment = range(candidate_up, candidate_down)
                peaks_in_segment = [x in candidate_segment for x in locs]
                is_valid_candidate = np.any(peaks_in_segment)
                if is_valid_candidate:
                    starts.append(candidate_up)
                    ends.append(candidate_down)
                    activity[candidate_segment] = 1.0
        else:
            starts = []
            ends = []
            activity = np.zeros(
                len(x),
            )
    else:
        starts = []
        ends = []
        activity = np.zeros(
            len(x),
        )
                            
    starts = np.array(starts)
    ends = np.array(ends)
    return activity, starts, ends

def main(conf):

    if os.path.isdir(conf["dataset_path"]):
        conf["save_dir"] = os.path.join(conf["dataset_path"],'filtered')
        os.makedirs(conf["save_dir"], exist_ok=True)
    else:
        print("dataset_path is not valid.\n")
        sys.exit(1)

    ### yamnet
    with tf.device('/CPU:0'):
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = model.class_map_path().numpy()
        class_names = class_names_from_csv(class_map_path)
        bird_ids = [i for i,c in enumerate(class_names) if c in CLASSES.values()]


        md = pd.read_csv(os.path.join(conf["dataset_path"],'separation','metadata.csv'))
        filtered_md = pd.DataFrame(columns=["timestamp","time","ID_left","ID_right","sep_path","bird","noise","left_csv","right_csv"])
        
        for index, row in md.iterrows():
            audio = []
            mini_dict = {'Left':row['ID_left'],'Right':row['ID_right']}
            if not os.path.exists(row["sep_path"]+'-'+row['ID_left']+'-source4.wav') or not os.path.exists(row["sep_path"]+'-'+row['ID_right']+'-source4.wav') :
                continue
            for i,c in enumerate(mini_dict):
                p = mini_dict[c]
                for s in range(1,NSOURCES+1):
                    input_mix,sample_rate = librosa.load(row["sep_path"]+'-'+p+'-source'+str(s)+'.wav',sr=None)
                    if  i==0 and s==1:
                        audio = np.zeros((2,NSOURCES,len(input_mix)))
                        stats = np.zeros((2,NSOURCES,2))
                    if len(input_mix) < len(audio[i,s-1,:]):
                        audio[i,s-1,:len(input_mix)] = input_mix
                    elif len(input_mix) > len(audio[i,s-1,:]):
                        audio[i,s-1,:] = input_mix[:len(audio[i,s-1,:])]
                    else:
                        audio[i,s-1,:] = input_mix
                    waveform = librosa.util.normalize(input_mix) 
                    scores, _, _ = model(waveform)
                    scores_np = scores.numpy()
                    scores_birds = scores_np[:,bird_ids]
                    stats[0,s-1,i] = np.max(scores_birds)
                    stats[1,s-1,i] = np.mean(scores_birds)
            
            ### at least one prediction is above threshold or its mean is above threshold (more than a single vocalization)
            if np.any(np.logical_and(stats[0,:,:]>THRESHOLD_MAX, stats[1,:,:]>THRESHOLD_MEAN)):
                ### avoid having bird predictions in all of the tracks - cases where mixit fails
                if np.all((np.logical_and(stats[0,:,:]>THRESHOLD_MAX, stats[1,:,:]>THRESHOLD_MEAN)).sum(axis=0)<NSOURCES-1):
                    pred_bird = np.zeros((2, len(audio[0,0,:])))
                    pred_noise = np.zeros((2, len(audio[0,0,:])))
                    for i in range(2):
                        max_max = stats[0,:,i].max() 
                        argmax_max = stats[0,:,i].argmax()
                        for s in range(NSOURCES):
                            if stats[0,s,i] == max_max:
                                pred_bird[i,:] += audio[i,s,:]
                            else:
                                pred_noise[i,:] += audio[i,s,:]
                    if WIENER:
                        pred_bird, pred_noise = wiener_filter(pred_bird, pred_noise)            
                    sf.write(os.path.join(conf["save_dir"],'{}-{}-{}-bird.wav'.format(row['timestamp'], row['ID_left'], row['ID_right'])), pred_bird.transpose(), sample_rate, 'PCM_24')
                    sf.write(os.path.join(conf["save_dir"],'{}-{}-{}-noise.wav'.format(row['timestamp'], row['ID_left'], row['ID_right'])), pred_noise.transpose(), sample_rate, 'PCM_24')
                    
                    frames2time = 512/sample_rate
                    rms = librosa.feature.rms(S=librosa.magphase(librosa.stft(pred_bird, n_fft=2048, hop_length=512, win_length=2048, window=np.ones, center=False))[0]).squeeze()
                    rms[0] = librosa.util.normalize(rms[0]) 
                    rms[1] = librosa.util.normalize(rms[1]) 
                    rms = scipy.ndimage.uniform_filter1d(rms, size=5)
                    activity, start, end = [], [], []
                    for i in range(2):
                        a, s, e = threshold_activity(rms[i], 0.01 , 0.1)
                        activity.append(a)
                        start.append(s-5) # -5 to compensate for the 5 frames window smoothing
                        end.append(e-5)

                    fig, ax = plt.subplots(figsize=(10,6))
                    ax.plot(rms[0], label='left',color='green')
                    ax.plot(rms[1], label='right',color='red')
                    colors = ['green','red']

                    birds = [row['ID_left'],row['ID_right']]
                    channels = ['left','right']
                    for i in range(2):
                        bird_events = []
                        for s,e in zip(start[i], end[i]):
                            #search with searchsorted for a overlapping interval in i-1 
                            sidx = np.searchsorted(start[i-1], s)
                            eidx = np.searchsorted(end[i-1], e)
                            if sidx == eidx:
                                intersect = 0
                                if sidx< len(start[i-1]) and start[i-1][sidx] < e and end[i-1][sidx] > s:
                                    intersect = min(end[i-1][sidx], e) - max(start[i-1][sidx], s) 
                                if eidx<len(end[i-1]) and start[i-1][eidx] < e and end[i-1][eidx] > s:
                                    intersect = min(end[i-1][eidx], e) - max(start[i-1][eidx], s) 
                                if intersect:
                                    print('overlap {} {} {} {} {} {} {}'.format(s,e,start[i-1][sidx],end[i-1][sidx],start[i-1][eidx],end[i-1][eidx], intersect))
                                    #import pdb;pdb.set_trace()
                                    if 2*intersect<(e-s) or rms[i][s:e].mean() > rms[i-1][s:e].mean() or np.abs(rms[i][s:e].max() - rms[i-1][s:e].max()) < 0.2: #this thresholding might be tricky
                                        bird_events.append([np.round(s*frames2time,2),np.round(e*frames2time,2),rms[i][s:e].mean()])
                                        ax.axvspan(s, e, alpha=0.2, color=colors[i])
                                else:
                                    bird_events.append([np.round(s*frames2time,2),np.round(e*frames2time,2),rms[i][s:e].mean()])
                                    ax.axvspan(s, e, alpha=0.2, color=colors[i])

                        with open(os.path.join(conf["save_dir"],'{}-{}-{}.csv'.format(row['timestamp'],channels[i],birds[i])), 'w') as f:
                            write = csv.writer(f)
                            write.writerow(['start','end','rms'])
                            write.writerows(bird_events)

                    plt.legend()
                    plt.show()
                    # import pdb;pdb.set_trace()

                    rowf = [row['time'],row['timestamp'], row['ID_left'], row['ID_right'], row["sep_path"],'{}-{}-{}-bird.wav'.format(row['timestamp'], row['ID_left'], row['ID_right']),'{}-{}-{}-noise.wav'.format(row['timestamp'], row['ID_left'], row['ID_right']),'{}-{}-{}.csv'.format(row['timestamp'],'left',birds[0]),'{}-{}-{}.csv'.format(row['timestamp'],'right',birds[1])]
                    filtered_md.loc[len(filtered_md)] = rowf
                else:
                    print("Skipping "+row["sep_path"]+" because mixit was unable to separate (all sources contain birds).")
            else:
                print("Skipping "+row["sep_path"]+" because there are no bird predictions above threshold in any of the separations.")

    filtered_md.to_csv(os.path.join(conf["save_dir"],'metadata.csv'), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    main(arg_dic)
