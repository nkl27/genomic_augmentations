import os, sys
import subprocess
import argparse
import h5py
import numpy as np
import pandas as pd
import torch
import logomaker
import matplotlib.pyplot as plt
import pathlib

def str2bool(v):
    # Sourced from <https://stackoverflow.com/a/43357954>
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args_train():
    """parse args for training of GRIM or GRIMA model (see train.py)"""
    
    # Set up parsing of command line arguments
    parser = argparse.ArgumentParser(description="GRIM version v0.2", 
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", metavar="/PATH/TO/DATA/", type=str,
                        help="path to HDF5 file containing data, pre-split into training, validation, and test sets")
    parser.add_argument("-expt_name", metavar="EXPERIMENT_NAME", type=str, 
                        help="name of experiment that will be used as file prefix", 
                        default="Expt")
    parser.add_argument("-output_dir", metavar="/PATH/TO/OUTPUT/", type=str, 
                        help="directory where output files will be stored; experiment name will be appended to results files", 
                        default="./Expt/")

    parser.add_argument("-d", metavar="NUM_FILTERS", type=int,
                        help="number of first layer convolutional filters to use", 
                        default=128)
    parser.add_argument("-p", metavar="DIM_Y", type=int,
                        help="dimensionality of the global encoding y = E_psi(x)", 
                        default=32)

    parser.add_argument("-encoder_type", metavar="ENCODER_TYPE", type=str,
                        help="encoder architecture type", 
                        default="CNN_S_noBN")
    parser.add_argument("-S", metavar="MAXPOOL_SIZE", type=int,
                        help="first max-pooling layer kernel size in encoder", 
                        default=4)
    parser.add_argument("-m", metavar="CONV_KERNEL_SIZE", type=int,
                        help="first convolutional layer filter length (kernel size) in encoder; currently only to be used with argument \"-encoder_type\" as \"DeepSEA\")", 
                        default=19)
    parser.add_argument("-encoder_lr", metavar="LEARNING_RATE", type=float,
                        help="learning rate for encoder", 
                        default=1e-4)
    parser.add_argument("-encoder_wd", metavar="WEIGHT_DECAY", type=float,
                        help="weight decay (L2 penalty) for encoder", 
                        default=1e-6)

    parser.add_argument("-alpha", metavar="ALPHA", type=float,
                        help="hyperparameter for weighing global MI estimator in objective", 
                        default=0.0)
    parser.add_argument("-beta", metavar="BETA", type=float,
                        help="hyperparameter for weighing local MI estimator in objective", 
                        default=1.0)
    parser.add_argument("-gamma", metavar="GAMMA", type=float,
                        help="hyperparameter for weighing prior matching term in objective", 
                        default=0.0)

    parser.add_argument("-global_disc_type", metavar="GLOBAL_DISC", type=str,
                        help="global discriminator architecture type; only used when alpha > 0", 
                        default="MaxpoolIMavg")
    parser.add_argument("-local_disc_type", metavar="LOCAL_DISC", type=str,
                        help="local discriminator architecture type; only used when beta > 0", 
                        default="IMmax")
    parser.add_argument("-prior_disc_type", metavar="PRIOR_DISC", type=str,
                        help="prior discriminator architecture type; only used when gamma > 0", 
                        default="default")
    parser.add_argument("-objective_lr", metavar="LEARNING_RATE", type=float,
                        help="learning rate for objective function (i.e., all discriminators used in objective)", 
                        default=1e-4)
    parser.add_argument("-objective_wd", metavar="WEIGHT_DECAY", type=float,
                        help="weight decay (L2 penalty) for objective function (i.e., all discriminators used in objective)", 
                        default=1e-6)
    
    parser.add_argument("-alphabet", metavar="ALPHABET", type=str, 
                        help="order of nucleotide channels in one-hot encoding of sequences x (e.g., \"ATCG\"), if custom", 
                        default="ACGT")
    parser.add_argument("-batch_size", metavar="BATCH_SIZE", type=int, 
                        help="batch size used in training", 
                        default=128)
    parser.add_argument("-epochs", metavar="MAX_EPOCHS", type=int, 
                        help="number of epochs to train", 
                        default=400)

    parser.add_argument("-scale_factor", metavar="SCALE_FACTOR", type=int,
                        help="factor by which to increase channels by depthwise conv (i.e., depthwise multiplier); only used with local discriminators \"cnc\", \"end\", \"elind\", \"eno\", \"ncf\"", 
                        default=2)
    parser.add_argument("-dim_emb", metavar="DIM_EMB", type=int,
                        help="dimensionality of further encoding/embedding of y and each c_i; only used with local discriminator \"ncf\"", 
                        default=32)
    
    parser.add_argument("-neg_samples", metavar="NEG_TYPE", type=str,
                        help="type of negative sample sequences x' used for GRIM mutual information estimation; options are \"dinuc\" (dinucleotide shuffled sequences), \"shuff\" (single-nucleotide shuffled sequences) and \"batch\" (other sequences x in batch; currently only to be used with argument \"-mi_estimator_type\" as \"infoNCE\")", 
                        default="dinuc")
    parser.add_argument("-mi_estimator_type", metavar="EST_TYPE", type=str,
                        help="type of mutual information estimator to be used; options are \"JSD\" (Jensen-Shannon Divergence-based estimator) and \"infoNCE\" (infoNCE-based estimator; currently only to be used with argument \"-neg_samples\" as \"batch\")", 
                        default="JSD")
    
    parser.add_argument("-augs", metavar="AUGMENTATION_STRING", type=str,
                        help="string specifying augmentations to use, comma delimited; possible augmentations are \"invert_seqs\", \"invert_rc_seqs\", \"delete_seqs\", \"roll\", \"roll_seqs\", \"insert\", \"insert_seqs\", \"rc\", \"mutate_seqs\", \"noise_gauss\" (e.g., \"noise_gauss,rc,roll_seqs,insert_seqs\")", 
                        default="")

    parser.add_argument("-invert_min", metavar="INVERT_MIN", type=int,
                        help="in inversion augmentations, minimum length of inversion", 
                        default=100)
    parser.add_argument("-invert_max", metavar="INVERT_MAX", type=int,
                        help="in inversion augmentations, maximum length of inversion", 
                        default=200)
    parser.add_argument("-delete_min", metavar="DELETE_MIN", type=int,
                        help="in deletion augmentations, minimum length of deletion", 
                        default=100)
    parser.add_argument("-delete_max", metavar="DELETE_MAX", type=int,
                        help="in deletion augmentations, maximum length of deletion", 
                        default=200)
    parser.add_argument("-shift_min", metavar="SHIFT_MIN", type=int,
                        help="in roll augmentations, minimum number of places by which position can be shifted", 
                        default=100)
    parser.add_argument("-shift_max", metavar="SHIFT_MAX", type=int,
                        help="in roll augmentations, maximum number of places by which position can be shifted", 
                        default=200)
    parser.add_argument("-insert_min", metavar="INSERT_MIN", type=int,
                        help="in insertion augmentations, minimum length of insertion", 
                        default=100)
    parser.add_argument("-insert_max", metavar="INSERT_MAX", type=int,
                        help="in insertion augmentations, maximum length of insertion", 
                        default=200)
    parser.add_argument("-rc_prob", metavar="RC_PROB", type=float,
                        help="in reverse complement augmentation, probability for each sequence to be \"mutated\" to its reverse complement", 
                        default=1.0)
    parser.add_argument("-mutate_frac", metavar="mutate_frac", type=float,
                        help="in random mutation augmentation, fraction of each sequence's nucleotides to mutate", 
                        default=0.1)
    parser.add_argument("-std", metavar="STDEV", type=float,
                        help="in Gaussian noise addition augmentation, standard deviation of Gaussian distribution from which noise is drawn", 
                        default=0.1)

    parser.add_argument("-filter_viz_subset", metavar="SIZE", type=int,
                        help="size of subset of test set to use for filter visualization; default value of None results in usage of whole test set", 
                        default=None)
    
    # Parse args
    args = parser.parse_args()
    
    # Add objective_type (for GRIM Config) to args for convenience
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    if (alpha > 0 and beta > 0 and gamma > 0):
        objective_type = "glm"
    elif (alpha > 0 and beta > 0 and gamma == 0):
        objective_type = "gl"
    elif (alpha > 0 and beta == 0 and gamma > 0):
        objective_type = "gm"
    elif (alpha == 0 and beta > 0 and gamma > 0):
        objective_type = "lm"
    elif (alpha > 0 and beta == 0 and gamma == 0):
        objective_type = "g"
    elif (alpha == 0 and beta > 0 and gamma == 0):
        objective_type = "l"
    else:
        raise ValueError("combination of alpha, beta, and gamma hyperparameters is invalid")
    args.objective_type = objective_type
    
    # Add sequence length L to args for convenience
    with h5py.File(args.data_path, 'r') as dataset:
        x_train = torch.from_numpy(np.array(dataset["X_train"]).astype(np.float32))
    args.L = x_train.shape[2] # length L of sequences
    
    return args


def parse_args_tomtom():
    """parse args for running Tomtom analysis on learned filters from GRIM model
        (see tomtom.py)
    """
    
    # Set up parsing of command line arguments
    parser = argparse.ArgumentParser(description="GRIM version v0.2", 
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("meme_path", metavar="/PATH/TO/MEME/FILE/", type=str,
                        help="path to MEME-format file containing alignments generated from trained GRIM model filters")
    parser.add_argument("db_path", metavar="/PATH/TO/DATABASE/", type=str,
                        help="path to MEME-format database of motifs against which to compare filters")
    
    parser.add_argument("-expt_name", metavar="EXPERIMENT_NAME", type=str, 
                        help="name of experiment that will be used as file prefix; recommended to be the same experiment name as used in training GRIM model", 
                        default="Expt")
    parser.add_argument("-output_dir", metavar="/PATH/TO/OUTPUT/", type=str, 
                        help="directory where output files will be stored; experiment name will be appended to results files", 
                        default="./Expt/")
    
    parser.add_argument('-match', metavar='MATCH_FLAG', type=str2bool, 
                    help='boolean specifying whether matches between Tomtom hits and known ground truth motifs should be quantified; if set to True, \"-ppms_path\" option must be set to the correct path for the pickled file containing PWMs from which the filters MEME file was generated', 
                    default=True)
    parser.add_argument("-ppms_path", metavar="/PATH/TO/PPMS/FILE/", type=str, 
                        help="path to the pickled (.p) file containing PWMs from which the filters MEME file was generated", 
                        default=None)
    
    # Parse args
    args = parser.parse_args()
    if args.match:
        assert args.ppms_path is not None, "\"-match\" option was set to True but \"-ppms_path\" option was set to None"
    
    return args


def mkdir(directory):
    """make directory"""
    if not os.path.isdir(directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print("Making directory: " + directory)


def make_directory(path, foldername, verbose=1):
    """make a directory"""
    if not os.path.isdir(path):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print("Making directory: " + path)

    outdir = os.path.join(path, foldername)
    if not os.path.isdir(outdir):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print("Making directory: " + outdir)
    
    return outdir    
    

def plot_filters(W, fig, num_cols=8, alphabet="ACGT", names=None, fontsize=12):
    """plot first-layer convolutional filters from PWM"""
    
    if alphabet == "ATCG":
        W = W[:,:,[0,2,3,1]]
    
    num_filter, filter_len, A = W.shape
    num_rows = np.ceil(num_filter/num_cols).astype(int)

    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for n, w in enumerate(W):
        ax = fig.add_subplot(num_rows,num_cols,n+1)

        # Calculate sequence logo heights -- information
        I = np.log2(4) + np.sum(w * np.log2(w+1e-7), axis=1, keepdims=True)
        logo = I*w
        
        # Create DataFrame for logomaker
        counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(filter_len)))
        for a in range(A):
            for l in range(filter_len):
                counts_df.iloc[l,a] = logo[l,a]
        
        logomaker.Logo(counts_df, ax=ax)
        ax = plt.gca()
        ax.set_ylim(0, 2) # set y-axis of all sequence logos to run from 0 to 2 bits
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])

        if names:
            plt.ylabel(names[n], fontsize=fontsize)
        
    return fig


def activation_pwm(fmap, X, threshold=0.5, window=20):
    # Set the left and right window sizes
    window_left = int(window/2)
    window_right = window - window_left

    N, L, A = X.shape # assume this ordering (i.e., TensorFlow ordering) of channels in X
    num_filters = fmap.shape[-1]

    W = []
    for filter_index in range(num_filters):

        # Find regions above threshold
        coords = np.where(fmap[:,:,filter_index] > np.max(fmap[:,:,filter_index])*threshold)
        x, y = coords

        # Sort score
        index = np.argsort(fmap[x,y,filter_index])[::-1]
        data_index = x[index].astype(int)
        pos_index = y[index].astype(int)

        # Make a sequence alignment centered about each activation (above threshold)
        seq_align = []
        for i in range(len(pos_index)):

            # Determine position of window about each filter activation
            start_window = pos_index[i] - window_left
            end_window = pos_index[i] + window_right

            # Check to make sure positions are valid
            if (start_window > 0) & (end_window < L):
                seq = X[data_index[i], start_window:end_window, :]
                seq_align.append(seq)

        # Calculate position probability matrix
        if len(seq_align) > 0:
            W.append(np.mean(seq_align, axis=0))
        else:
            W.append(np.zeros((window, A)))
    W = np.array(W)

    return W


def clip_filters(W, threshold=0.5, pad=3):
    W_clipped = []
    for w in W:
        L, A = w.shape
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index)-pad, 0)
            end = np.minimum(np.max(index)+pad+1, L)
            W_clipped.append(w[start:end,:])
        else:
            W_clipped.append(w)

    return W_clipped


def generate_meme(W, output_file='meme.txt', prefix='Filter'):
    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C  %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j, pwm in enumerate(W):
        if np.count_nonzero(pwm) > 0:
            L, A = pwm.shape
            f.write('MOTIF %s%d \n' % (prefix, j))
            f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
            for i in range(L):
                f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
            f.write('\n')
    
    f.close()


def tomtom(motif_path, db_path, output_path, evalue=False, thresh=0.5, dist='pearson', png=None, tomtom_path='tomtom'):
    """perform Tomtom analysis"""
    "dist: allr | ed | kullback | pearson | sandelin"
    
    cmd = [tomtom_path, '-thresh', str(thresh), '-dist', dist]
    if evalue:
        cmd.append('-evalue')  
    if png:
        cmd.append('-png')
    cmd.extend(['-oc', output_path, motif_path, db_path])
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr


def count_meme_entries(motif_path):
    """Count number of meme entries"""
    with open(motif_path, 'r') as f:
        counter = 0
        for line in f:
            if line[:6] == 'letter':
                counter += 1
    return counter


def match_hits_to_ground_truth(file_path, motifs, motif_names=None, num_filters=32):
    """Works with Tomtom version 5.1.0 
    inputs:
        - file_path: .tsv file output from tomtom analysis
        - motifs: list of list of JASPAR ids
        - motif_names: name of motifs in the list
        - num_filters: number of filters in conv layer (needed to normalize -- tomtom doesn't always give results for every filter)
    outputs:
        - match_fraction: fraction of total filters (num_filters) that match ground truth motifs
        - match_any: fraction of total filters (num_filters) that match any motif in JASPAR (except Gremb1)
        - filter_match: for each filter, the name of best hit (if matched to a ground truth motif)
        - filter_qvalue: the q-value of the best hit to a ground truth motif (1.0 means no hit)
        - motif_qvalue: for each ground truth motif, the best q-value hit (0.0 means no hit)
        - motif_counts: for each ground truth motif, the number of filter hits
    """
    
    # add a zero for indexing no hits
    motifs = motifs.copy()
    motif_names = motif_names.copy()
    motifs.insert(0, [''])
    motif_names.insert(0, '')

    # get dataframe for tomtom results
    df = pd.read_csv(file_path, delimiter='\t')

    # loop through filters
    filter_qvalue = np.ones(num_filters)
    best_match = np.zeros(num_filters).astype(int)
    correction = 0  
    for name in np.unique(df['Query_ID'][:-3].to_numpy()):
        filter_index = int(name.split('r')[1])

        # get tomtom hits for filter
        subdf = df.loc[df['Query_ID'] == name]
        targets = subdf['Target_ID'].to_numpy()

        # loop through ground truth motifs
        for k, motif in enumerate(motifs): 

            # loop through variations of ground truth motif
            for id in motif: 

                # check if there is a match
                index = np.where((targets == id) == True)[0]
                if len(index) > 0:
                    qvalue = subdf['q-value'].to_numpy()[index]

                    # check to see if better motif hit, if so, update
                    if filter_qvalue[filter_index] > qvalue:
                        filter_qvalue[filter_index] = qvalue
                        best_match[filter_index] = k 

        # don't count hits to Gmeb1 (because too many)
        index = np.where((targets == 'MA0615.1') == True)[0]
        if len(index) > 0:
            if len(targets) == 1:
                correction += 1

    # get names of best match motifs
    filter_match = [motif_names[i] for i in best_match]

    # get hits to any motif
    num_matches = len(np.unique(df['Query_ID'])) - 3.0 # 3 is correction because of last 3 lines of comments in the tsv file (may change across tomtom versions)
    match_any = (num_matches - correction) / num_filters # counts hits to any motif (not including Grembl)

    # match fraction to ground truth motifs
    match_index = np.where(filter_qvalue != 1.)[0]
    if any(match_index):
        match_fraction = len(match_index)/float(num_filters)
    else:
        match_fraction = 0.0

    # get the number of hits and minimum q-value for each motif
    num_motifs = len(motifs) - 1
    motif_qvalue = np.zeros(num_motifs)
    motif_counts = np.zeros(num_motifs)
    for i in range(num_motifs):
        index = np.where(best_match == i+1)[0]
        if len(index) > 0:
            motif_qvalue[i] = np.min(filter_qvalue[index])
            motif_counts[i] = len(index)

    return match_fraction, match_any, filter_match, filter_qvalue, motif_qvalue, motif_counts


def motif_comparison_synthetic_datasets(file_path, num_filters=32, custom_motifs=None, custom_motif_names=None):
    """Compares tomtom analysis for filters trained on synthetic multitask classification.
        Works with Tomtom version 5.1.0.
    inputs:
        - file_path: .tsv file output from tomtom analysis
        - num_filters: number of filters in conv layer (needed to normalize -- tomtom doesn't always give results for every filter)
    outputs:
        - match_fraction: fraction of total filters (num_filters) that match ground truth motifs
        - match_any: fraction of total filters (num_filters) that match any motif in JASPAR (except Gremb1)
        - filter_match: for each filter, the name of best hit (if matched to a ground truth motif)
        - filter_qvalue: the q-value of the best hit to a ground truth motif (1.0 means no hit)
        - motif_qvalue: for each ground truth motif, the best q-value hit (0.0 means no hit)
        - motif_counts: for each ground truth motif, the number of filter hits
    """

    # arid3 = ['MA0151.1', 'MA0601.1', 'PB0001.1']
    cebpb = ['MA0466.1', 'MA0466.2']
    fosl1 = ['MA0477.1']
    gabpa = ['MA0062.1', 'MA0062.2']
    mafk = ['MA0496.1', 'MA0496.2']
    max1 = ['MA0058.1', 'MA0058.2', 'MA0058.3']
    mef2a = ['MA0052.1', 'MA0052.2', 'MA0052.3']
    nfyb = ['MA0502.1', 'MA0060.1', 'MA0060.2']
    sp1 = ['MA0079.1', 'MA0079.2', 'MA0079.3']
    srf = ['MA0083.1', 'MA0083.2', 'MA0083.3']
    stat1 = ['MA0137.1', 'MA0137.2', 'MA0137.3', 'MA0660.1', 'MA0773.1']
    yy1 = ['MA0095.1', 'MA0095.2']

    motifs = [cebpb, fosl1, gabpa, mafk, max1, mef2a, nfyb, sp1, srf, stat1, yy1] # NB! excludes arid3
    motif_names = ['CEBPB', 'FOSL1', 'GABPA', 'MAFK', 'MAX', 'MEF2A', 'NFYB', 'SP1', 'SRF', 'STAT1', 'YY1']
    
    # Use below for synthetic one-motif doped datasets (and derivatives thereof)
    motifs = [cebpb, fosl1, mafk, max1, sp1, srf]
    motif_names = ['CEBPB', 'FOSL1', 'MAFK', 'MAX', 'SP1', 'SRF']
    
    if (custom_motifs is not None) and (custom_motif_names is not None):
        motifs = custom_motifs
        motif_names = custom_motif_names
    
    match_fraction, match_any, filter_match, filter_qvalue, min_qvalue, motif_counts = match_hits_to_ground_truth(file_path, motifs, motif_names=motif_names, num_filters=num_filters)

    return match_fraction, match_any, filter_match, filter_qvalue, min_qvalue, motif_counts