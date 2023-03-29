import os, sys
import argparse
import tqdm
import subprocess
import pandas as pd
import numpy as np
from io import StringIO
import seaborn as sns
import collections
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection


from contextlib import redirect_stdout



parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path to spanish-carrion-crow dataset"
)

parser.add_argument(
    "--db_dir",
    type=str,
    default="",
    help="Directory where the eval results will be stored. Leave empty to use the same as input.",
)

parser.add_argument(
    "--audfprint_path", type=str, required=True, help="Local path to audfprint repository"
)

parser.add_argument(
    "--density",
    type=int,
    default=20,
    required=False,
    help="Density of the peaks for audfprint",
)

parser.add_argument(
    "--stage",
    default="fingerprint",
    const="fingerprint",
    nargs="?",
    choices=["fingerprint", "match"],
)


def compute_fingerprint(pair, **kwargs):
    subdir, filename = pair
    assert os.path.exists(os.path.join(kwargs['db_dir'],'db'+str(kwargs['density'])+'.pklz'))
    command_name = ['audfprint','add']
 
    command_name.extend(
        ["-d", os.path.join(kwargs["db_dir"], "db" + str(kwargs["density"]) + ".pklz")]
    )
    command_name.extend(["-n", kwargs["density"]])
    command_name.append(os.path.join(subdir, filename))
    print(command_name)

    audfprint.main(command_name)


def compute_match(pair, **kwargs):

    subdir, filename = pair

    command_name = [
        "audfprint",
        "match",
        "--dbase",
        os.path.join(kwargs["db_dir"], "db" + str(kwargs["density"]) + ".pklz"),
    ]
    command_name.append(os.path.join(subdir, filename))
    command_name.extend(["-n", kwargs["density"]])
    command_name.append("--find-time-range")
    # import pdb;pdb.set_trace()
    f = StringIO()
    with redirect_stdout(f):
        audfprint.main(command_name)
    out = f.getvalue().splitlines()

    # result = subprocess.run(command_name, stdout=subprocess.PIPE, universal_newlines=True)

    column_names = [
        "Query",
        "Query begin time",
        "Query end time",
        "Reference",
        "Reference begin time",
        "Reference end time",
        "Confidence",
    ]
    df_out = pd.DataFrame(columns=column_names)
    if "NOMATCH" not in out[3]:
        print(out[3])
        # 'Matched   34.9 s starting at   16.6 s in /home/marius/data/BAF_sep/LibriMSD_Waveunet_config/query_1037-music.wav
        # to time   98.5 s in /home/marius/data/BAF/audios/references/ref_1695.wav with     9 of   324 common hashes at rank  0
        for i in range(3, len(out) - 1):
            # import pdb;pdb.set_trace()
            times = [t.split(" ")[-1] for t in out[i].split(" s ")]
            paths = [p.split(" ")[0] for p in out[i].split(" in ")]
            cs = out[i].split(" common hashes ")[0].split("with")[-1].split(" of ")
            df_out.loc[filename + str(i)] = pd.Series(
                {
                    "Query": os.path.basename(paths[1]).replace(".wav", ""),
                    "Query begin time": float(times[1]),
                    "Query end time": float(times[1]) + float(times[2]),
                    "Reference": os.path.basename(paths[2]).replace(".wav", ""),
                    "Reference begin time": float(times[3]),
                    "Reference end time": float(times[3]) + float(times[2]),
                    "Confidence": float(cs[0]) / float(cs[1]),
                }
            )
    else:
        print('NO MATCHES')

    return df_out


def main(conf):

    if os.path.isdir(conf["dataset_path"]):
        conf["subdataset"] = 'AL'
        conf["individual"] = 'Naranja'
        conf["database_files_path"] = os.path.join(conf["dataset_path"],'separation','19',conf["subdataset"], conf["individual"])
        conf["input_files_path"] = os.path.join(conf["dataset_path"],'19',conf["subdataset"], conf["individual"])
        conf["annotation_path"] = os.path.join(conf["dataset_path"],'labelled_datasets','individual_vocalizations','classification_with_file.csv')
        conf["save_dir"] = os.path.join(conf["dataset_path"],'fingerprints','audfprint','19',conf["subdataset"], conf["individual"])
        os.makedirs(conf["save_dir"], exist_ok=True)
    else:
        print("dataset_path is not valid.\n")
        sys.exit(1)

    sys.path.append(conf["audfprint_path"])
    global audfprint
    import audfprint

    filelist = []
    subdirs = {}
    for subdir, dirs, files in os.walk(conf["input_files_path"]):
        fileshere = [
            (subdir, filename) for filename in files if filename.endswith(".wav")
        ]
        subdirs[subdir] = {}
        filelist.extend(fileshere)

    
    filelist_train, filelist_test = sklearn.model_selection.train_test_split(filelist,
        random_state=42, 
        test_size=0.8, 
        shuffle=True)
    
   


    if conf["stage"] == "fingerprint":
        for sd, f in filelist_train:
            subdirs[sd][f] = 1
        dbfilelist = []
        for subdir, dirs, files in os.walk(conf["database_files_path"]):
            fileshere = [
                (subdir, filename) for filename in files 
                if filename.endswith(".wav") and subdir.replace('separation'+os.path.sep,'') in subdirs and filename.split('-')[0]+'.wav' in subdirs[subdir.replace('separation'+os.path.sep,'')]
            ]
            dbfilelist.extend(fileshere)
        command_name = ["audfprint", "new"]
        command_name.extend(
            ["-d", os.path.join(conf["db_dir"], "db" + str(conf["density"]) + ".pklz")]
        )
        command_name.extend(["-n", conf["density"]])
        command_name.append(os.path.join(dbfilelist[0][0], dbfilelist[0][1]))
        audfprint.main(command_name)
        for pair in tqdm.tqdm(dbfilelist):
            compute_fingerprint(pair, **conf)

    elif conf["stage"] == "match":
        for sd, f in filelist_train:
            subdirs[sd][f] = 1
        dbfilelist = []
        for subdir, dirs, files in os.walk(conf["database_files_path"]):
            fileshere = [
                (subdir, filename) for filename in files 
                if filename.endswith(".wav") and subdir.replace('separation'+os.path.sep,'') in subdirs and filename.split('-')[0]+'.wav' in subdirs[subdir.replace('separation'+os.path.sep,'')]
            ]
            dbfilelist.extend(fileshere)
        #load annotations from csv
        df = pd.read_csv(conf["annotation_path"])

        i = 0
        column_names = [
            "Query",
            "Query begin time",
            "Query end time",
            "Reference",
            "Reference begin time",
            "Reference end time",
            "Confidence",
        ]
        df = pd.DataFrame(columns=column_names)
        for pair in tqdm.tqdm(dbfilelist):
            match_df = compute_match(pair, **conf)
            #import pdb;pdb.set_trace()
            df = df.append(match_df)
            i += 1
            if i > 1e8:
                break

      
if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    main(arg_dic)

#########
# python audfprint_fingerprint.py --stage fingerprint --audfprint_path /home/marius/code/audfprint/ --dataset_path /home/marius/data/spanish-carrion-crows/
# python audfprint_fingerprint.py --stage match --audfprint_path /home/marius/code/audfprint/ --dataset_path /home/marius/data/spanish-carrion-crows/
# python audfprint_fingerprint.py --stage match --audfprint_path /home/marius/code/audfprint/ --dataset_path  /home/marius/data/spanish-carrion-crows/
