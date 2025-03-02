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

import matplotlib.pyplot as plt

import sklearn
from sklearn import cluster, pipeline


import tensorflow as tf
import tensorflow_hub as hub

from contextlib import redirect_stdout

NSOURCES = 4
CLS_YAMNET = {97:"Turkey",98:"Gobble",99:"Duck",100:"Quack",101:"Goose",106:"Bird",107:"Bird vocalization, bird call, bird song",108:"Chirp, tweet",109:"Squawk",111:"Coo",112:"Crow",113:"Caw",127:"Frog"}

PLOT = False

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path to zebra finch Audio folder containing Left and Right sub-folders"
)
parser.add_argument(
    "--classifier",
    choices=["yamnet"], ### perch does not work so well here so we exclude it for now
    default="yamnet",
    help="Method to use for classification"
)

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'ebird2021' in row:
                class_names.append(row['ebird2021'])
            elif 'display_name' in row:
                class_names.append(row['display_name'])
            else:
                raise ValueError('No column named "ebird2021" or "display_name" found in the class map CSV.')

    return class_names

def ensure_sample_rate(waveform, original_sample_rate,
                        desired_sample_rate=32000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        waveform = librosa.resample(waveform, orig_sr=original_sample_rate, target_sr=desired_sample_rate)
    return desired_sample_rate, waveform

def frame_audio(
        audio_array: np.ndarray,
        window_size_s: float = 5.0,
        hop_size_s: float = 5.0,
        sample_rate = 32000,
    ) -> np.ndarray:
    """Helper function for framing audio for inference."""
    if window_size_s is None or window_size_s < 0:
        return audio_array[np.newaxis, :]
    frame_length = int(window_size_s * sample_rate)
    hop_length = int(hop_size_s * sample_rate)
    framed_audio = tf.signal.frame(audio_array, frame_length, hop_length, pad_end=True)
    return framed_audio

def get_onset(rms, rms_other, index, window=10, threshold=0.2):
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
    if index > onset:
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
    if index < offset:
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
        if conf["classifier"] == "yamnet":
            model = hub.load('https://tfhub.dev/google/yamnet/1')
            class_map_path = model.class_map_path().numpy()
            class_names = class_names_from_csv(class_map_path)
            bird_ids = [i for i,c in enumerate(class_names) if c in CLS_YAMNET.values()]
            sample_rate_cls = 16000
            threshold_max = 0.1
            threshold_mean = 0.001
        elif conf["classifier"] == "perch":
            model = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/3")
            labels_path = hub.resolve("https://tfhub.dev/google/bird-vocalization-classifier/3") + "/assets/label.csv"
            class_names = class_names_from_csv(labels_path)
            bird_ids = [i for i,c in enumerate(class_names) if c=='zebfin2']
            sample_rate_cls = 32000
            threshold_max = 0.1
            threshold_mean = 0.001
        else:
            raise ValueError("classifier must be either yamnet or perch")


        md = pd.read_csv(os.path.join(conf["dataset_path"],'separation','metadata.csv'))
        filtered_md = pd.DataFrame(columns=["timestamp","time","ID_left","ID_right","sep_path","bird","noise","left_csv","right_csv"])
        
        md_vox = pd.DataFrame(columns=['fn','audio_fp', 'selection_table_fp'])
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
                    waveform =  np.nan_to_num(waveform)
                    _, input_mix_4pred = ensure_sample_rate(waveform, sample_rate, sample_rate_cls)
                
                    if hasattr(model, "infer_tf"):
                        fixed_tm = frame_audio(input_mix_4pred)
                        all_logits, all_embeddings = model.infer_tf(fixed_tm[:1])
                        scores = []
                        for window in fixed_tm[1:]:
                            logits, _ = model.infer_tf(window[np.newaxis, :])
                            probabilities = tf.nn.softmax(logits)
                            all_logits = np.concatenate([all_logits, logits], axis=0)
                            scores.append(probabilities.numpy().squeeze())
                        # frame = 0
                        # for frame_logits in all_logits:
                        #     probabilities = tf.nn.softmax(frame_logits)
                        #     argmax = np.argmax(probabilities)
                        #     print(f"For frame {frame}, the audio is from the class {class_names[argmax]} (element:{argmax} in the label.csv file), with probability of {probabilities[argmax]}")
                        #     frame += 1
                    else:
                        scores, embeddings, _ = model(input_mix_4pred)
                        scores = scores.numpy()

                    scores_np = np.array(scores)
                    scores_birds = scores_np[:,bird_ids]
                    stats[0,s-1,i] = np.max(scores_birds)
                    stats[1,s-1,i] = np.mean(scores_birds)
            # f0, voiced_flag, voiced_probs = librosa.pyin(audio.sum(axis=0), fmin=200,fmax=4000, sr=sample_rate, frame_length=2048, hop_length=512, fill_na=0, center=False,switch_prob=0.1,no_trough_prob=0.01)
            
            # print(voiced_probs.max(axis=1))
            max_probs = stats[0].max(axis=1)
            maxp = max_probs.max()
            if PLOT:
                print(row["sep_path"])
                print(np.round(max_probs,3))
                print(stats[0])
            
            ### at least one prediction is above threshold or its mean is above threshold (one or more sources contain bird sounds)
            if np.any(np.logical_and(stats[0,:,:]>threshold_max, stats[1,:,:]>threshold_mean)):
                pred_bird = np.zeros((2, len(audio[0,0,:])))
                pred_noise = np.zeros((2, len(audio[0,0,:])))

                if max_probs.argmax() < 2:
                    pred_bird+= audio[:,0,:]
                    if (max_probs[1] / max_probs[0]) > 0.6:
                        pred_bird+= audio[:,1,:]
                    else:
                        pred_noise+= audio[:,1,:]
                    pred_noise+= audio[:,2,:]
                    pred_noise+= audio[:,3,:]
                    # for i in range(2):
                    #     for s in range(NSOURCES):
                    #         if max_probs[s]/maxp > 0.8:
                    #             pred_bird[i,:] += audio[i,s,:]
                    #         else:
                    #             pred_noise[i,:] += audio[i,s,:] 
                    
                
                    ### eliminate noise from bird track
                    f0, voiced_flag, voiced_probs = librosa.pyin(pred_bird.sum(axis=0), fmin=200,fmax=2000, sr=sample_rate, frame_length=2048, hop_length=512, fill_na=0, center=False,switch_prob=0.1,no_trough_prob=0.01)
                    rms = librosa.feature.rms(S=librosa.magphase(librosa.stft(pred_bird.sum(axis=0), n_fft=2048, hop_length=512, win_length=2048, window=np.ones, center=False))[0]).squeeze()
                    rms = librosa.util.normalize(rms) 
                    # f0, voiced_flag, voiced_probs = librosa.pyin(pred_bird, fmin=200,fmax=4000, sr=sample_rate, frame_length=2048, hop_length=512, fill_na=0, center=False,switch_prob=0.1,no_trough_prob=0.01)
                    # rms = librosa.feature.rms(S=librosa.magphase(librosa.stft(pred_bird, n_fft=2048, hop_length=512, win_length=2048, window=np.ones, center=False))[0]).squeeze()
                    # rms[0] = librosa.util.normalize(rms[0]) 
                    # rms[1] = librosa.util.normalize(rms[1]) 
                    rms =  np.nan_to_num(rms)
                    voiced_probs =  np.nan_to_num(voiced_probs)
                    #mask = np.logical_and(np.logical_or(voiced_probs[0]>0.04, voiced_probs[1]>0.04),np.logical_or(rms[0]>0.04, rms[1]>0.04))
                    mask = np.logical_and(voiced_probs>0.04, rms>0.04)
                    mask_samples=np.array([[any(mask[i-2:i+6])]*512 for i in range(len(mask))]).flatten()[0:len(pred_bird.sum(axis=0))]
                    mask_samples = list(mask_samples)
                    if len(mask_samples) < len(pred_bird.sum(axis=0)):
                        mask_samples.extend([mask_samples[-1]]*(len(pred_bird.sum(axis=0))-len(mask_samples)))
                    mask_samples = np.array(mask_samples)
                    pred_noise = pred_noise + pred_bird*~mask_samples
                    pred_bird = pred_bird*mask_samples
                    
                    ### normalize
                    pred_bird =  np.nan_to_num(pred_bird/(np.max(np.abs(pred_bird))+1e-8))

                    frames2time = 512/sample_rate
                    spec = librosa.magphase(librosa.stft(pred_bird, n_fft=2048, hop_length=512, win_length=2048, window=np.ones, center=True))[0]
                    rms = librosa.feature.rms(S=spec).squeeze()
                    rms =  np.nan_to_num(rms)
                    # rms = scipy.signal.savgol_filter(rms, 3, 2) # window size 3, polynomial order 2
                    ### compute peaks in both channels
                    peaks0, _ = scipy.signal.find_peaks(rms[0], height=0.015, distance=6) 
                    peaks1, _ = scipy.signal.find_peaks(rms[1], height=0.015, distance=6)
                    
                    max_peak_rate = 10 ### every 10 * 512 samples we have a peak 
                    max_peaks = int(len(rms[0])/max_peak_rate)
                    if len(peaks0)<max_peaks and len(peaks1)<max_peaks:
                        
                        ### save
                        sf.write(os.path.join(conf["save_dir"],'{}-{}-{}-bird.wav'.format(row['timestamp'], row['ID_left'], row['ID_right'])), pred_bird.transpose(), sample_rate, 'PCM_24')
                        sf.write(os.path.join(conf["save_dir"],'{}-{}-{}-noise.wav'.format(row['timestamp'], row['ID_left'], row['ID_right'])), pred_noise.transpose(), sample_rate, 'PCM_24')
                    
                        if PLOT:
                            fig, ax = plt.subplots(figsize=(10,6))
                            ax.plot(rms[0], label='left',color='green')
                            ax.plot(rms[1], label='right',color='red')
                            ax.plot(peaks0, rms[0,peaks0], "o", color="green" )
                            ax.plot(peaks1, rms[1,peaks1], "x", color="red" )
                        
                        colors = ['green','red']
                        birds = [row['ID_left'],row['ID_right']]
                        channels = ['left','right']
                        bird_events = [[],[]]
                        ### save diarization as Raven
                        md_raven = pd.DataFrame(columns=['Selection','View','Channel','Begin Time (s)','End Time (s)','Low Freq (Hz)','High Freq (Hz)','Annotation'])
                        
                        
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
                                md_raven.loc[len(md_raven)] = [len(md_raven)+1,'Waveform 1',0,np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),200,6000,'Left']
                                if PLOT: ax.axvspan(onset, offset, alpha=0.5, color=colors[0])
                                idxp0 += 1
                                birdl = True
                            elif idxp0==len(peaks0):
                                ### no more left peaks
                                onset = get_onset(rms[1], rms[0], peaks1[idxp1],10)
                                offset = get_offset(rms[1], rms[0], peaks1[idxp1],10)
                                bird_events[1].append([np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),rms[1][peaks1[idxp1]]])
                                md_raven.loc[len(md_raven)] = [len(md_raven)+1,'Waveform 1',1,np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),200,6000,'Right']
                                if PLOT: ax.axvspan(onset, offset, alpha=0.5, color=colors[1])
                                idxp1 += 1
                                birdr = True
                            elif np.abs(peaks0[idxp0] - peaks1[idxp1])<3: ### we account for inter-mic delay
                                ### we have a vocalization in both of the channels (same peak)
                                ### compute harmonic salience 
                                harms = [1, 2, 3, 4]
                                weights = [1.0, 0.5, 0.33, 0.25]
                                freqs = librosa.fft_frequencies(sr=sample_rate)
                                sal = librosa.salience(spec[:,:,peaks0[idxp0]-6:peaks0[idxp0]+6].mean(axis=0), freqs=freqs, harmonics=harms, weights=weights, fill_value=0)
                                sal0 = librosa.salience(spec[0,:,peaks0[idxp0]-6:peaks0[idxp0]+6], freqs=freqs, harmonics=harms, weights=weights, fill_value=0)
                                sal1 = librosa.salience(spec[1,:,peaks1[idxp1]-6:peaks1[idxp1]+6], freqs=freqs, harmonics=harms, weights=weights, fill_value=0)
                                if PLOT: print(peaks0[idxp0], np.round(np.corrcoef(librosa.util.normalize(sal1).sum(axis=1),librosa.util.normalize(sal0).sum(axis=1))[0,1],2), np.round(sal.mean(),2), np.round(sal0.mean(),2), np.round(sal1.mean(),2))
                                #if np.corrcoef(librosa.util.normalize(sal1).sum(axis=1),librosa.util.normalize(sal0).sum(axis=1))[0,1] < 0.5 and sal.mean()>0.4 and (np.maximum(sal1.mean(),sal0.mean())/np.minimum(sal0.mean(),sal1.mean()) < 2):
                                if np.corrcoef(librosa.util.normalize(sal1).sum(axis=1),librosa.util.normalize(sal0).sum(axis=1))[0,1] < 0.55 and sal.mean()>0.2 and (np.maximum(sal1.mean(),sal0.mean())/np.minimum(sal0.mean(),sal1.mean()) < 2) and sal0.mean()>0.15 and sal1.mean() < 0.15:
                                    ##### both birds vocalize at the same time
                                    onset = get_onset(rms[0], rms[1], peaks0[idxp0],10)
                                    offset = get_offset(rms[0], rms[1], peaks0[idxp0],10)
                                    bird_events[0].append([np.round((onset+1)*frames2time,2),np.round(offset*frames2time,2),rms[0][peaks0[idxp0]]])
                                    md_raven.loc[len(md_raven)] = [len(md_raven)+1,'Waveform 1',0,np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),200,6000,'Overlap']
                                    if PLOT: ax.axvspan(onset, offset, alpha=0.5, color=colors[0])
                                    onset = get_onset(rms[1], rms[0], peaks1[idxp1],10)
                                    offset = get_offset(rms[1], rms[0], peaks1[idxp1],10)
                                    bird_events[1].append([np.round((onset+1)*frames2time,2),np.round(offset*frames2time,2),rms[1][peaks1[idxp1]]])
                                    if PLOT: ax.axvspan(onset, offset, alpha=0.5, color=colors[1])
                                    birdr = True
                                    birdl = True
                                else:
                                    if rms[0][peaks0[idxp0]] > rms[1][peaks1[idxp1]]:
                                        ### left bird vocalizes
                                        onset = get_onset(rms[0], rms[1], peaks0[idxp0],10)
                                        offset = get_offset(rms[0], rms[1], peaks0[idxp0],10)
                                        bird_events[0].append([np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),rms[0][peaks0[idxp0]]])
                                        md_raven.loc[len(md_raven)] = [len(md_raven)+1,'Waveform 1',0,np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),200,6000,'Left']
                                        if PLOT: ax.axvspan(onset, offset, alpha=0.5, color=colors[0])
                                        birdl = True
                                    else:
                                        ### right bird vocalizes
                                        onset = get_onset(rms[1], rms[0], peaks1[idxp1],10)
                                        offset = get_offset(rms[1], rms[0], peaks1[idxp1],10)
                                        bird_events[1].append([np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),rms[1][peaks1[idxp1]]])
                                        md_raven.loc[len(md_raven)] = [len(md_raven)+1,'Waveform 1',1,np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),200,6000,'Right']
                                        if PLOT: ax.axvspan(onset, offset, alpha=0.5, color=colors[1])
                                        birdr = True
                                idxp0 += 1
                                idxp1 += 1
                            elif peaks0[idxp0] < peaks1[idxp1]:
                                ### left bird vocalizes in left channel only
                                onset = get_onset(rms[0], rms[1], peaks0[idxp0],10)
                                offset = get_offset(rms[0], rms[1], peaks0[idxp0],10)
                                bird_events[0].append([np.round((onset+1)*frames2time,2),np.round(offset*frames2time,2),rms[0][peaks0[idxp0]]])
                                md_raven.loc[len(md_raven)] = [len(md_raven)+1,'Waveform 1',0,np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),200,6000,'Left']
                                if PLOT: ax.axvspan(onset, offset, alpha=0.5, color=colors[0])
                                idxp0 += 1
                                birdl = True
                            elif peaks0[idxp0] > peaks1[idxp1]:
                                ### right bird vocalizes in right channel only
                                onset = get_onset(rms[1], rms[0], peaks1[idxp1],10)
                                offset = get_offset(rms[1], rms[0], peaks1[idxp1],10)
                                bird_events[1].append([np.round((onset+1)*frames2time,2),np.round(offset*frames2time,2),rms[1][peaks1[idxp1]]])
                                md_raven.loc[len(md_raven)] = [len(md_raven)+1,'Waveform 1',1,np.round((onset+1)*frames2time,2),np.round((offset+1)*frames2time,2),200,6000,'Right']
                                if PLOT: ax.axvspan(onset, offset, alpha=0.5, color=colors[1])
                                idxp1 += 1
                                birdr = True

                
                        ### save detection/diarization results
                        for idx in range(2):
                            with open(os.path.join(conf["save_dir"],'{}-{}-{}.csv'.format(row['timestamp'],channels[idx],birds[idx])), 'w') as f:
                                write = csv.writer(f)
                                write.writerow(['start','end','rms'])
                                write.writerows(bird_events[idx])
                        md_raven.to_csv(os.path.join(conf["save_dir"],'{}.Table.1.selections.txt'.format(row['timestamp'])), index=False)
                        if PLOT:       
                            plt.title(row['sep_path'])
                            plt.legend()
                            plt.show()

                        rowf = [row['time'],row['timestamp'], row['ID_left'], row['ID_right'], row["sep_path"],'{}-{}-{}-bird.wav'.format(row['timestamp'], row['ID_left'], row['ID_right']),'{}-{}-{}-noise.wav'.format(row['timestamp'], row['ID_left'], row['ID_right']),'{}-{}-{}.csv'.format(row['timestamp'],'left',birds[0]),'{}-{}-{}.csv'.format(row['timestamp'],'right',birds[1])]
                        filtered_md.loc[len(filtered_md)] = rowf
                        ### save voxaboxen files
                        md_vox.loc[len(md_vox)] = ['{}-{}-{}-bird.wav'.format(row['timestamp'], row['ID_left'], row['ID_right']), os.path.join(conf["save_dir"],'{}-{}-{}-bird.wav'.format(row['timestamp'], row['ID_left'], row['ID_right'])), os.path.join(conf["save_dir"],'{}.Table.1.selections.txt'.format(row['timestamp']))]
                    else:
                        print("Skipping "+row["sep_path"]+" because too many peaks were detected.")
                else:
                    print("Skipping "+row["sep_path"]+" because the bird predominant channel is 3 or 4 (mixit likely fails).")

            else:
                print("Skipping "+row["sep_path"]+" because there are no bird predictions above threshold in any of the separations.")

    filtered_md.to_csv(os.path.join(conf["save_dir"],'metadata.csv'), index=False)
    md_vox.to_csv(os.path.join(conf["dataset_path"],'voxaboxen.csv'), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    main(arg_dic)
