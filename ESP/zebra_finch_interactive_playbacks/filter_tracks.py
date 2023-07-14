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
import bisect

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

# def denoising(x: np.ndarray, threshold: float) -> np.ndarray:
#     """Denoises the array using an estimated Gaussian noise distribution.

#     Forms a noise estimate by a) estimating mean+std, b) removing extreme
#     values, c) re-estimating mean+std for the noise, and then d) classifying
#     values in the spectrogram as 'signal' or 'noise' based on likelihood under
#     the revised estimate. We then apply a mask to return the signal values.

#     Args:
#         melspec: input array of rank 2.
#         threshold: z-score theshold for separating signal from noise. On the first
#             pass, we use 2 * threshold, and on the second pass we use threshold
#             directly.

#     Returns:
#         The denoised x.
#     """

#     feature_mean = np.mean(x, axis=1, keepdims=True)
#     feature_std = np.std(x, axis=1, keepdims=True)
#     is_noise = (x - feature_mean) < 2 * threshold * feature_std

#     noise_counts = np.sum(is_noise.astype(x.dtype), axis=1, keepdims=True)
#     noise_mean = np.sum(x * is_noise, axis=0, keepdims=True) / (noise_counts + 1)
#     noise_var = np.sum(
#         is_noise * np.square(x - noise_mean), axis=1, keepdims=True
#     )
#     noise_std = np.sqrt(noise_var / (noise_counts + 1))

#     # Recompute signal/noise separation.
#     demeaned = x - noise_mean
#     is_signal = demeaned >= threshold * noise_std
#     is_signal = is_signal.astype(x.dtype)
#     is_noise = 1.0 - is_signal

#     signal_part = is_signal * x
#     noise_part = is_noise * noise_mean
#     reconstructed = signal_part + noise_part - noise_mean
#     return reconstructed

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

# def threshold_activity(x, Tp, Ta):
#     locs = scipy.signal.find_peaks(x,height = Tp)[0]
#     if len(locs) > 1:
#         y = (x > Ta) * 1
#         act = np.diff(y)
#         u = np.where(act == 1)[0]
#         d = np.where(act == -1)[0]
#         signal_length = len(x)
#         if len(u) > 0 and len(d) > 0:
#             if d[0] < u[0]:
#                 u = np.insert(u, 0, 0)
                
#             if d[-1] < u[-1]:
#                 d = np.append(d, signal_length-1)
                
#             starts = []
#             ends = []
            
#             activity = np.zeros(signal_length,)
            
#             for candidate_up, candidate_down in zip(u, d):
#                 candidate_segment = range(candidate_up, candidate_down)
#                 peaks_in_segment = [x in candidate_segment for x in locs]
#                 is_valid_candidate = np.any(peaks_in_segment)
#                 if is_valid_candidate:
#                     starts.append(candidate_up)
#                     ends.append(candidate_down)
#                     activity[candidate_segment] = 1.0
#         else:
#             starts = []
#             ends = []
#             activity = np.zeros(
#                 len(x),
#             )
#     else:
#         starts = []
#         ends = []
#         activity = np.zeros(
#             len(x),
#         )
                            
#     starts = np.array(starts)
#     ends = np.array(ends)
#     return activity, starts, ends

def get_onset(rms, rms_other, index, window=10, threshold=0.4):
    range_start = max(0, index - window)
    threshold_attack = threshold * rms[index]
    onsets = np.where(rms[range_start:index] < threshold_attack)
    if len(onsets[0]) == 0:
        local_minimas, _ = scipy.signal.find_peaks(-rms[range_start:index])
        if len(local_minimas) > 0:
            return local_minimas[-1] + index - window
        else:
            return range_start
    onset = onsets[0][-1] + index - window
    offset_before = np.where(rms_other[onset+1:index] > rms[onset+1:index])
    if len(offset_before[0]) > 0:
        offset = onset + 1 + offset_before[0][-1] 
    return onset

def get_offset(rms, rms_other, index, window, threshold=0.1):
    range_end = min(len(rms), index + window)
    threshold_release = threshold * rms[index]
    offsets = np.where(rms[index:range_end] < threshold_release)
    if len(offsets[0]) == 0:
        local_minimas, _ = scipy.signal.find_peaks(-rms[index:range_end])
        if len(local_minimas) > 0:
            return local_minimas[0] + index
        else:
            return range_end
    offset = offsets[0][0] + index
    onset_after = np.where(rms_other[index:offset-1] > rms[index:offset-1])
    if len(onset_after[0]) > 0:
        offset = offset - 1 - onset_after[0][0]
    return offset

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
            
            ### at least one prediction is above threshold or its mean is above threshold (one or more sources contain bird sounds)
            if np.any(np.logical_and(stats[0,:,:]>THRESHOLD_MAX, stats[1,:,:]>THRESHOLD_MEAN)):
                ### avoid having bird predictions in all of the tracks - cases where mixit fails
                if np.all((np.logical_and(stats[0,:,:]>THRESHOLD_MAX, stats[1,:,:]>THRESHOLD_MEAN)).sum(axis=0)<NSOURCES):
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
                    
                    ### eliminate noise from bird track
                    f0, voiced_flag, voiced_probs = librosa.pyin(pred_bird.sum(axis=0), fmin=200,fmax=3000, sr=sample_rate, frame_length=2048, hop_length=512, fill_na=0, center=False,switch_prob=0.1,no_trough_prob=0.01)
                    rms = librosa.feature.rms(S=librosa.magphase(librosa.stft(pred_bird.sum(axis=0), n_fft=2048, hop_length=512, win_length=2048, window=np.ones, center=False))[0]).squeeze()
                    rms = librosa.util.normalize(rms) 
                    mask = np.logical_and(voiced_probs>0.05, rms>0.05)
                    mask_samples=np.array([[any(mask[i-2:i+6])]*512 for i in range(len(mask))]).flatten()[0:len(pred_bird.sum(axis=0))]
                    mask_samples = list(mask_samples)
                    if len(mask_samples) < len(pred_bird.sum(axis=0)):
                        mask_samples.extend([mask_samples[-1]]*(len(pred_bird.sum(axis=0))-len(mask_samples)))
                    mask_samples = np.array(mask_samples)
                    #import pdb;pdb.set_trace()
                    pred_noise = pred_noise + pred_bird*~mask_samples
                    pred_bird = pred_bird*mask_samples
                    # fig, ax = plt.subplots(figsize=(10,6))
                    # ax.plot(rms, label='rms',color='green')
                    # ax.plot(voiced_probs, label='prob',color='blue')
                    # plt.title(row['sep_path'])
                    # plt.legend()
                    # plt.show()
                    sf.write(os.path.join(conf["save_dir"],'{}-{}-{}-bird.wav'.format(row['timestamp'], row['ID_left'], row['ID_right'])), pred_bird.transpose(), sample_rate, 'PCM_24')
                    sf.write(os.path.join(conf["save_dir"],'{}-{}-{}-noise.wav'.format(row['timestamp'], row['ID_left'], row['ID_right'])), pred_noise.transpose(), sample_rate, 'PCM_24')
                    
                    ### normalize
                    pred_bird = pred_bird/np.max(np.abs(pred_bird))

                    frames2time = 512/sample_rate
                    spec = librosa.magphase(librosa.stft(pred_bird, n_fft=2048, hop_length=512, win_length=2048, window=np.ones, center=True))[0]
                    rms = librosa.feature.rms(S=spec).squeeze()
                    # rms = librosa.util.normalize(rms) 
                    rms =  np.nan_to_num(rms)
                    peaks0, _ = scipy.signal.find_peaks(rms[0], height=0.02)
                    peaks1, _ = scipy.signal.find_peaks(rms[1], height=0.02)
                    # rms = denoising(rms, 0.8)
                    # rms[0] = scipy.ndimage.uniform_filter1d(rms[0], 5)
                    # rms[1] = scipy.ndimage.uniform_filter1d(rms[1], 5)
                    # on = librosa.onset.onset_strength(S=librosa.feature.melspectrogram(y=pred_bird, sr=sample_rate, n_mels=128, fmin=1000, fmax=10000, n_fft=2048, hop_length=512, win_length=2048, center=True))
                    # on = librosa.util.normalize(on) 
                    # on =  np.nan_to_num(on)
                    # import pdb; pdb.set_trace()
                    # activity, start, end = [], [], []
                    # for i in range(2):
                    #     a, s, e = threshold_activity(librosa.util.normalize(rms[i]), 0.01 , 0.1)
                    #     activity.append(a)
                    #     start.append(s) 
                    #     end.append(e)
                

                    fig, ax = plt.subplots(figsize=(10,6))
                    ax.plot(rms[0], label='left',color='green')
                    ax.plot(rms[1], label='right',color='red')
                    ax.plot(peaks0, rms[0,peaks0], "o", color="green" )
                    ax.plot(peaks1, rms[1,peaks1], "x", color="red" )
                    # ax.plot(f00/np.max(np.abs(f0)), label='left',color='yellow')
                    # ax.plot(f01/np.max(np.abs(f0)), label='right',color='cyan')
                    # ax.plot(f0/np.max(np.abs(f0)), label='f0',color='black')
                    # ax.plot(voiced_probs, label='prob',color='blue')
                    # ax.plot(on[0], label='lefto',color='yellow')
                    # ax.plot(on[1], label='righto',color='cyan')
                    colors = ['green','red']

                    birds = [row['ID_left'],row['ID_right']]
                    channels = ['left','right']
                    i=0
                    j=0
                    bird_events = [[],[]]
                    ### determine which bird is vocalizing

                    idxp0 = 0
                    idxp1 = 0
                    while idxp0<len(peaks0) or idxp1<len(peaks1):
                        birdl = False
                        birdr = False
                        if idxp1==len(peaks1):
                            ### no more right peaks
                            onset = get_onset(rms[0], rms[1], peaks0[idxp0],10)
                            offset = get_offset(rms[0], rms[1], peaks0[idxp0],10)
                            bird_events[0].append([np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),rms[0][peaks0[idxp0]]])
                            ax.axvspan(onset, offset, alpha=0.5, color=colors[0])
                            idxp0 += 1
                            birdl = True
                        elif idxp0==len(peaks0):
                            ### no more left peaks
                            onset = get_onset(rms[1], rms[0], peaks1[idxp1],10)
                            offset = get_offset(rms[1], rms[0], peaks1[idxp1],10)
                            bird_events[1].append([np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),rms[1][peaks1[idxp1]]])
                            ax.axvspan(onset, offset, alpha=0.5, color=colors[1])
                            idxp1 += 1
                            birdr = True
                        elif np.abs(peaks0[idxp0] - peaks1[idxp1])<3: ### we account for inter-mic delay
                            ### we have a vocalization in both of the channels (same peak)
                            # diff = np.abs(librosa.util.normalize(c[0])-librosa.util.normalize(c[1]))
                            # print(diff.mean())
                            harms = [1, 2, 3, 4]
                            weights = [1.0, 0.5, 0.33, 0.25]
                            freqs = librosa.fft_frequencies(sr=sample_rate)
                            sal = librosa.salience(spec[:,:,peaks0[idxp0]-6:peaks0[idxp0]+6].mean(axis=0), freqs=freqs, harmonics=harms, weights=weights, fill_value=0)
                            sal0 = librosa.salience(spec[0,:,peaks0[idxp0]-6:peaks0[idxp0]+6], freqs=freqs, harmonics=harms, weights=weights, fill_value=0)
                            sal1 = librosa.salience(spec[1,:,peaks1[idxp1]-6:peaks1[idxp1]+6], freqs=freqs, harmonics=harms, weights=weights, fill_value=0)
                            # import pdb; pdb.set_trace()
                            # start0 = np.maximum(0, (peaks0[idxp0]-6)*512)
                            # end0 = np.minimum(len(pred_bird[0]), (peaks0[idxp0]+6)*512)
                            # start1 = np.maximum(0, (peaks1[idxp1]-6)*512)
                            # end1 = np.minimum(len(pred_bird[1]), (peaks1[idxp1]+6)*512)
                            # f00, _, _ = librosa.pyin(librosa.util.normalize(pred_bird[0,start0:end0]), boltzmann_parameter=2, fmin=200,fmax=3000, sr=sample_rate, frame_length=2048, hop_length=512, fill_na=0, center=True,switch_prob=0.05,no_trough_prob=0.01)
                            # f01, _, _ = librosa.pyin(librosa.util.normalize(pred_bird[1,start1:end1]), boltzmann_parameter=2, fmin=200,fmax=3000, sr=sample_rate, frame_length=2048, hop_length=512, fill_na=0, center=True,switch_prob=0.05,no_trough_prob=0.01)
                            # ax.plot(np.arange(np.floor(start0/512),np.floor(start0/512)+len(f00)),f00/np.max(np.abs(f0)),color='yellow')
                            # ax.plot(np.arange(np.floor(start1/512),np.floor(start1/512)+len(f01)),f01/np.max(np.abs(f0)),color='cyan')

                            # # coeff=np.corrcoef(spec[0,:,peaks0[idxp0]-6,peaks0[idxp0]+6],spec[1,:,peaks1[idxp1]-6:peaks1[idxp1]+6])
                            # # cos = scipy.spatial.distance.cdist(spec[0,:,peaks0[idxp0]-6:peaks0[idxp0]+6].T,spec[1,:,peaks1[idxp1]-6:peaks1[idxp1]+6].T,'cosine')
                            # cxy = np.nan_to_num(np.corrcoef(spec[0,:,peaks0[idxp0]-6:peaks0[idxp0]+6].T,spec[1,:,peaks1[idxp1]-6:peaks1[idxp1]+6].T))
                            # cxx = np.nan_to_num(np.corrcoef(spec[0,:,peaks0[idxp0]-6:peaks0[idxp0]+6].T,spec[0,:,peaks0[idxp0]-6:peaks0[idxp0]+6].T))
                            # cyy = np.nan_to_num(np.corrcoef(spec[0,:,peaks1[idxp1]-6:peaks1[idxp1]+6].T,spec[1,:,peaks1[idxp1]-6:peaks1[idxp1]+6].T))
                            # coherence = np.abs(cxy)**2/(np.abs(cxx)*np.abs(cyy)+1e-8)
                            # print(peaks0[idxp0],coherence.mean())
                            # print(cxy.mean())
                            #import pdb; pdb.set_trace()
                            if np.corrcoef(librosa.util.normalize(sal1).sum(axis=1),librosa.util.normalize(sal0).sum(axis=1))[0,1] < 0.5 and sal.mean()>0.4 and (np.maximum(sal1.mean(),sal0.mean())/np.minimum(sal0.mean(),sal1.mean()) < 2):
                                ##### both birds vocalize at the same time
                                # print(peaks0[idxp0], np.round(np.corrcoef(librosa.util.normalize(sal1).sum(axis=1),librosa.util.normalize(sal0).sum(axis=1))[0,1],2), np.round(sal.mean(),2), np.round(sal0.mean(),2), np.round(sal1.mean(),2))
                                onset = get_onset(rms[0], rms[1], peaks0[idxp0],10)
                                offset = get_offset(rms[0], rms[1], peaks0[idxp0],10)
                                bird_events[0].append([np.round((onset+1)*frames2time,2),np.round(offset*frames2time,2),rms[0][peaks0[idxp0]]])
                                ax.axvspan(onset, offset, alpha=0.5, color=colors[0])
                                onset = get_onset(rms[1], rms[0], peaks1[idxp1],10)
                                offset = get_offset(rms[1], rms[0], peaks1[idxp1],10)
                                bird_events[1].append([np.round((onset+1)*frames2time,2),np.round(offset*frames2time,2),rms[1][peaks1[idxp1]]])
                                ax.axvspan(onset, offset, alpha=0.5, color=colors[1])
                                birdr = True
                                birdl = True
                            else:
                                if rms[0][peaks0[idxp0]] > rms[1][peaks1[idxp1]]:
                                    ### left bird vocalizes
                                    onset = get_onset(rms[0], rms[1], peaks0[idxp0],10)
                                    offset = get_offset(rms[0], rms[1], peaks0[idxp0],10)
                                    bird_events[0].append([np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),rms[0][peaks0[idxp0]]])
                                    # if len(bird_events[1])>0 and bird_events[0][-1][0] > bird_events[1][-1][1]:
                                    #     diff = int(abs(bird_events[0][-1][0] - bird_events[1][-1][1]) // 2)
                                    #     bird_events[0][-1][0] = bird_events[0][-1][0] - diff
                                    #     bird_events[1][-1][1] = bird_events[1][-1][1] - diff
                                    ax.axvspan(onset, offset, alpha=0.5, color=colors[0])
                                    birdl = True
                                else:
                                    ### right bird vocalizes
                                    onset = get_onset(rms[1], rms[0], peaks1[idxp1],10)
                                    offset = get_offset(rms[1], rms[0], peaks1[idxp1],10)
                                    bird_events[1].append([np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),rms[1][peaks1[idxp1]]])
                                    # if len(bird_events[0])>0 and bird_events[1][-1][0] > bird_events[0][-1][1]:
                                    #     diff = int(abs(bird_events[1][-1][0] - bird_events[0][-1][1]) // 2)
                                    #     bird_events[1][-1][0] = bird_events[1][-1][0] - diff
                                    #     bird_events[0][-1][1] = bird_events[0][-1][1] - diff
                                    ax.axvspan(onset, offset, alpha=0.5, color=colors[1])
                                    birdr = True
                            idxp0 += 1
                            idxp1 += 1
                        elif peaks0[idxp0] < peaks1[idxp1]:
                            ### left bird vocalizes in left channel only
                            onset = get_onset(rms[0], rms[1], peaks0[idxp0],10)
                            offset = get_offset(rms[0], rms[1], peaks0[idxp0],10)
                            bird_events[0].append([np.round((onset+1)*frames2time,2),np.round(offset*frames2time,2),rms[0][peaks0[idxp0]]])
                            # if len(bird_events[1])>0 and bird_events[0][-1][0] > bird_events[1][-1][1]:
                            #     diff = int(abs(bird_events[0][-1][0] - bird_events[1][-1][1]) // 2)
                            #     bird_events[0][-1][0] = bird_events[0][-1][0] - diff
                            #     bird_events[1][-1][1] = bird_events[1][-1][1] - diff
                            ax.axvspan(onset, offset, alpha=0.5, color=colors[0])
                            idxp0 += 1
                            birdl = True
                        elif peaks0[idxp0] > peaks1[idxp1]:
                            ### right bird vocalizes in right channel only
                            onset = get_onset(rms[1], rms[0], peaks1[idxp1],10)
                            offset = get_offset(rms[1], rms[0], peaks1[idxp1],10)
                            bird_events[1].append([np.round((onset+1)*frames2time,2),np.round(offset*frames2time,2),rms[1][peaks1[idxp1]]])
                            # if len(bird_events[0])>0 and bird_events[1][-1][0] > bird_events[0][-1][1]:
                            #     diff = int(abs(bird_events[1][-1][0] - bird_events[0][-1][1]) // 2)
                            #     bird_events[1][-1][0] = bird_events[1][-1][0] - diff
                            #     bird_events[0][-1][1] = bird_events[0][-1][1] - diff
                            ax.axvspan(onset, offset, alpha=0.5, color=colors[1])
                            idxp1 += 1
                            birdr = True

                    # if len(start[0]) == 0 and len(start[1]) == 0:
                    #     print("Skipping "+row["sep_path"]+" because no vocalizations were detected.")
                    #     plt.title(row['sep_path'])
                    #     plt.legend()
                    #     plt.show()
                    #     continue
                    # else:
                    #     while i < len(start[0]) or j < len(start[1]):
                    #         if j==len(start[1]):
                    #             bird_events[0].append([np.round(start[0][i]*frames2time,2),np.round(end[0][i]*frames2time,2),rms[0][start[0][i]:start[0][i]].mean()])
                    #             ax.axvspan(start[0][i], end[0][i], alpha=0.5, color=colors[0])
                    #             i += 1
                    #         elif i==len(start[0]):
                    #             bird_events[1].append([np.round(start[1][j]*frames2time,2),np.round(end[1][j]*frames2time,2),rms[1][start[1][j]:start[1][j]].mean()])
                    #             ax.axvspan(start[1][j], end[1][j], alpha=0.5, color=colors[1])
                    #             j += 1
                    #         else:
                    #             intersect = 0
                    #             if start[1][j] < end[0][i] and end[1][j] > start[0][i]:
                    #                 intersect = min(end[1][j], end[0][i]) - max(start[1][j], start[0][i]) 
                    #                 print("overlap: {}, start: {}, end: {}, start: {}, end {}".format(intersect, start[0][i], end[0][i], start[1][j], end[1][j]))
                    #                 idx_start0 = bisect.bisect_left(peaks0, start[0][i])
                    #                 idx_end0 = bisect.bisect_right(peaks0, end[0][i])
                    #                 idx_start1 = bisect.bisect_left(peaks1, start[1][j])
                    #                 idx_end1 = bisect.bisect_right(peaks1, end[1][j])
                    #                 local_peaks0 = peaks0[idx_start0:idx_end0]
                    #                 local_peaks1 = peaks1[idx_start1:idx_end1]

                                            
                    #                 # import pdb; pdb.set_trace()
                    #                 # #### both birds vocalizing almost at the same time
                    #                 # if intersect < (end[0][i] - start[0][i])//2 and intersect < (end[1][j] - start[1][j])//2:
                    #                 #     bird_events[0].append([np.round(start[0][i]*frames2time,2),np.round(end[0][i]*frames2time,2),rms[0][start[0][i]:start[0][i]].mean()])
                    #                 #     ax.axvspan(start[0][i], end[0][i], alpha=0.2, color=colors[1])
                    #                 #     bird_events[1].append([np.round(start[1][j]*frames2time,2),np.round(end[1][j]*frames2time,2),rms[1][start[1][j]:start[1][j]].mean()])
                    #                 #     ax.axvspan(start[1][j], end[1][j], alpha=0.2, color=colors[1])
                    #                 # #### one bird vocalizing and the other one is silent, which is which?
                    #                 # else:
                    #                 #     start0 = np.maximum(0, np.minimum(start[0][i]-1,start[1][j]-1)*512)
                    #                 #     end0 = np.minimum(len(pred_bird[0]), np.maximum(end[0][i]+1,end[1][j]+1)*512)
                    #                 #     interval0 = pred_bird[0, start0 : end0]
                    #                 #     interval1 = pred_bird[1, start0 : end0]
                    #                 #     delay = 256
                    #                 #     lm0, _ = scipy.signal.find_peaks(-rms[0][start[0][i]-5:end[0][i]+5])
                    #                 #     lm1, _ = scipy.signal.find_peaks(-rms[1][start[1][j]-5:end[1][j]+5])
                    #                 #     correlation = scipy.signal.correlate(interval1, interval0, mode="full")
                    #                 #     lags = scipy.signal.correlation_lags(interval0.size, interval1.size, mode="full")
                    #                 #     lag = lags[np.argmax(correlation)]
                    #                 #     print('lag {}'.format(lag)) 
                    #                 #     lm0 = lm0 + start[0][i]-5
                    #                 #     lm1 = lm1 + start[1][j]-5 
                    #                 #     ax.plot(lm0, rms[0,lm0], "o", color="green" )
                    #                 #     ax.plot(lm1, rms[1,lm1], "x", color="red" )
                    #                 #     if int(lag) > 0:
                    #                 #         bird_events[0].append([np.round(start[0][i]*frames2time,2),np.round(end[0][i]*frames2time,2),rms[0][start[0][i]:start[0][i]].mean()])
                    #                 #         ax.axvspan(start[0][i], end[0][i], alpha=0.2, color=colors[0])                           
                    #                 #     else:
                    #                 #         bird_events[1].append([np.round(start[1][j]*frames2time,2),np.round(end[1][j]*frames2time,2),rms[1][start[1][j]:start[1][j]].mean()])
                    #                 #         ax.axvspan(start[1][j], end[1][j], alpha=0.2, color=colors[1])    
                    #                 # #import pdb;pdb.set_trace()
                    #                 i += 1
                    #                 j += 1
                    #             else:
                    #                 if end[0][i] < start[1][j]:
                    #                     bird_events[0].append([np.round(start[0][i]*frames2time,2),np.round(end[0][i]*frames2time,2),rms[0][start[0][i]:start[0][i]].mean()])
                    #                     ax.axvspan(start[0][i], end[0][i], alpha=0.2, color=colors[1])
                    #                     i+=1
                    #                 else:
                    #                     bird_events[1].append([np.round(start[1][j]*frames2time,2),np.round(end[1][j]*frames2time,2),rms[1][start[1][j]:start[1][j]].mean()])
                    #                     ax.axvspan(start[1][j], end[1][j], alpha=0.2, color=colors[1])
                    #                     j+=1
                    for idx in range(2):
                        with open(os.path.join(conf["save_dir"],'{}-{}-{}.csv'.format(row['timestamp'],channels[idx],birds[idx])), 'w') as f:
                            write = csv.writer(f)
                            write.writerow(['start','end','rms'])
                            write.writerows(bird_events[idx])
                    plt.title(row['sep_path'])
                    plt.legend()
                    plt.show()

                    rowf = [row['time'],row['timestamp'], row['ID_left'], row['ID_right'], row["sep_path"],'{}-{}-{}-bird.wav'.format(row['timestamp'], row['ID_left'], row['ID_right']),'{}-{}-{}-noise.wav'.format(row['timestamp'], row['ID_left'], row['ID_right']),'{}-{}-{}.csv'.format(row['timestamp'],'left',birds[0]),'{}-{}-{}.csv'.format(row['timestamp'],'right',birds[1])]
                    filtered_md.loc[len(filtered_md)] = rowf
                else:
                    print("Skipping "+row["sep_path"]+" because mixit was unable to separate (all sources contain birds).")
                    import pdb; pdb.set_trace()
            else:
                print("Skipping "+row["sep_path"]+" because there are no bird predictions above threshold in any of the separations.")

    filtered_md.to_csv(os.path.join(conf["save_dir"],'metadata.csv'), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    main(arg_dic)
