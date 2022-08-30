import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ushuffle import shuffle

import logomaker
import functools
from itertools import chain, combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

def compose2(f, g):
    return lambda *args, **kwargs: f(g(*args, **kwargs))

def compose(*fs):
    return functools.reduce(compose2, fs) # NB! function listed *last* in fs becomes innermost function

def powerset_without_empty(iterable):
    "Use: powerset_without_empty([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def calculate_auroc(y_true, y_score):
    """Calculate AUROC for each individual class in data (usually multilabel binary);
        if AUROC is not defined for a particular class (e.g., because only one label
        is present in that class), np.nan is returned for that class. See
        <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html>
        for more details.
    
    Args:
      y_true: a numpy array of shape (n_samples, n_classes)
          True labels or binary label indicators.
      y_scores: a numpy array of shape (n_samples, n_classes)
          Target scores (e.g., outputs from a model). 

    Returns:
      (n_classes) array, where each element is the individual AUROC for that class (or np.nan).
    """
    
    num_classes = y_true.shape[-1]
    aurocs_by_class = []
    
    for clazz in range(num_classes):
        try:
            aurocs_by_class.append( roc_auc_score(y_true[:,clazz], y_score[:,clazz]) )
        except ValueError:
            aurocs_by_class.append( np.nan )
    
    return np.array(aurocs_by_class)


# def convert_one_hot(sequences, alphabet="ACGT") -> np.ndarray:
#     """Convert flat array of sequences to one-hot representation.

#     **Important**: all letters in `sequences` *must* be contained in `alphabet`, and
#     all sequences must have the same length.

#     Parameters
#     ----------
#     sequences : numpy.ndarray of strings
#         The array of strings. Should be one-dimensional.
#     alphabet : str
#         The alphabet of the sequences.

#     Returns
#     -------
#     Numpy array of sequences in one-hot representation. The shape of this array is
#     `(len(sequences), len(sequences[0]), len(alphabet))`.

#     Examples
#     --------
#     >>> one_hot(["TGCA"], alphabet="ACGT")
#     array([[[0., 0., 0., 1.],
#             [0., 0., 1., 0.],
#             [0., 1., 0., 0.],
#             [1., 0., 0., 0.]]])
#     """
#     sequences = np.asanyarray(sequences)
#     if sequences.ndim != 1:
#         raise ValueError("array of sequences must be one-dimensional.")
#     n_sequences = sequences.shape[0]
#     sequence_len = len(sequences[0])

#     # Unpack strings into 2D array, where each point has one character.
#     s = np.zeros((n_sequences, sequence_len), dtype="U1")
#     for i in range(n_sequences):
#         s[i] = list(sequences[i])

#     # Make an integer array from the string array.
#     pre_onehot = np.zeros(s.shape, dtype=np.uint8)
#     for i, letter in enumerate(alphabet):
#         # do nothing on 0 because array is initialized with zeros.
#         if i:
#             pre_onehot[s == letter] = i

#     # create one-hot representation
#     n_classes = len(alphabet)
#     return np.eye(n_classes)[pre_onehot]


def convert_onehot_to_sequence(one_hot, alphabet="ACGT"):
    """Convert DNA/RNA sequences from one-hot representation to
    string representation.

    Parameters
    ----------
    one_hot : <numpy.ndarray>
        one_hot encoded sequence with shape (N, L, A)
    alphabet : <str>
        DNA = 'ACGT'

    Returns
    -------
    sequences : <numpy.ndarray>
    A numpy vector of sequences in string representation.

    Example
    -------
    >>> one_hot = np.array(
            [[[1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]],

            [[0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.]]]
                )
    >>> sequences = convert_onehot_to_sequence(one_hot)
    >>> sequences
    array([['A', 'A', 'G', 'A', 'C'],
       ['T', 'C', 'G', 'C', 'A']], dtype=object)
    """
    assert alphabet in ["ACGT", "ACGU"], "Enter a valid alphabet"

    # convert alphabet to dictionary
    alphabet_dict = {i: a for i, a in enumerate(list(alphabet))}

    # get indices of one-hot
    seq_indices = np.argmax(one_hot, axis=2)  # (N, L)

    # convert index to sequence
    sequences = []
    for seq_index in seq_indices:
        seq = pd.Series(seq_index).map(alphabet_dict)
        sequences.append(seq)
    return sequences # np.asarray(sequences)


def convert_seq(seqs_onehot):
    """convert one-hot representation of sequences into actual string of nucleotides;
        assumes that input X comes in as numpy array with shape (N, A, L), where
        N is number of sequences, A = 4 is number of nucleotides, and L is length of sequences, 
        (in other words, L must be along axis=2, just like the output of convert_one_hot())"""
    seqs = []
    seqs_onehot_transpose = seqs_onehot.transpose([0,2,1]) # put length on axis=1 for compatibility with `OneHotEncoder` class from sklearn
    
    one_hot_encoder = OneHotEncoder(categories=[np.array(['A', 'C', 'G', 'T'])], sparse=False)
    one_hot_encoder.fit(np.array([['A'], ['C'], ['G'], ['T']]))
    
    for seq_onehot in seqs_onehot_transpose:
        seq = ''.join(one_hot_encoder.inverse_transform(seq_onehot).squeeze())
        seqs.append(seq)
    
    # Convert to numpy array
    seqs = np.array(seqs)
    
    return seqs


def convert_one_hot(sequence, max_length=None):
    """convert DNA/RNA sequences to a one-hot representation"""

    one_hot_seq = []
    for seq in sequence:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))
        index = [j for j in range(seq_length) if seq[j] == 'A']
        one_hot[0,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'C']
        one_hot[1,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'G']
        one_hot[2,index] = 1
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        one_hot[3,index] = 1

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length)/2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4,offset2))])

        one_hot_seq.append(one_hot)

    # convert to numpy array
    # one_hot_seq = np.array(one_hot_seq)

    return one_hot_seq


def convert_seq(seqs_onehot):
    """convert one-hot representation of sequences into actual string of nucleotides;
        assumes that input X comes in as numpy array with shape (N, A, L), where
        N is number of sequences, A = 4 is number of nucleotides, and L is length of sequences, 
        (in other words, L must be along axis=2, just like the output of convert_one_hot())"""
    seqs = []
    seqs_onehot_transpose = seqs_onehot.transpose([0,2,1]) # put length on axis=1 for compatibility with `OneHotEncoder` class from sklearn
    
    one_hot_encoder = OneHotEncoder(categories=[np.array(['A', 'C', 'G', 'T'])], sparse=False)
    one_hot_encoder.fit(np.array([['A'], ['C'], ['G'], ['T']]))
    
    for seq_onehot in seqs_onehot_transpose:
        seq = ''.join(one_hot_encoder.inverse_transform(seq_onehot).squeeze())
        seqs.append(seq)
    
    # Convert to numpy array
    # seqs = np.array(seqs)
    
    return seqs


def convert_targets(labels):
    """convert 1-D array of labels into 2-D array of targets 
        in one-hot target format (e.g., input of array([1, 0, 2])
        yields output of array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])) 
    """
    N = len(labels)
    max_labels = max(labels) + 1
    targets = np.zeros((N, max_labels))
    
    for i in range(N):
        targets[i, labels[i]] = 1
    
    return targets


def find_dinuc_shuff_fail_indices(seqs):
    """find indices of sequences in seqs array where dinucleotide shuffle fails 
        (i.e., produces the same input sequence)"""  
    dinuc_shuff_fail_indices = []
    
    for i, seq in enumerate(seqs):
        seq = seq.upper()
        seq_shuffled = shuffle( bytes(seq, "utf-8") , 2).decode("utf-8") # dinucleotide shuffle
        if seq_shuffled == seq:
            dinuc_shuff_fail_indices.append(i)
    
    return dinuc_shuff_fail_indices


def dinucleotide_shuffle(seqs):
    """shuffle each sequence in seqs while keeping dinucleotide frequency the same"""
    seqs_dinucleotide_shuffled = []
    
    for i, seq in enumerate(seqs):
        seq = seq.upper()
        seq_shuffled = shuffle( bytes(seq, "utf-8") , 2).decode("utf-8") # dinucleotide shuffle
        assert seq_shuffled != seq, "dinucleotide shuffle on sequence at index " + str(i) + " produced same sequence"
        
        seqs_dinucleotide_shuffled.append(seq_shuffled)
        
    seqs_dinucleotide_shuffled = np.array(seqs_dinucleotide_shuffled)
    
    return seqs_dinucleotide_shuffled


def kmerize(seqs, k=6):
    """convert DNA/RNA sequences to sets of k-mers"""
    seqs_kmers = []
    for seq in seqs:
        seq = seq.upper()
        seqs_kmers.append( [seq[i:i+k] for i in range(0, len(seq)-k+1)] )

    # convert to numpy array
    # one_hot_seq = np.array(one_hot_seq)

    return seqs_kmers


# def shuffle_onehot(one_hot, k=1):
#     """Shuffle one-hot represented sequences while preserving k-let frequencies.
    
#     Parameters
#     ----------
#     one_hot : numpy.ndarray
#         One_hot encoded sequence with shape (N, L, A)
#     k : int
#         k of k-let frequencies to preserve (e.g., with k = 2, dinucleotide
#         shuffle is performed); default is k = 1 (i.e., single-nucleotide
#         shuffle)

#     Returns
#     -------
#     Numpy array of one-hot represented shuffled sequences, of the same shape
#     as one_hot.

#     Examples
#     --------
#     >>> seqs = ["ACGT", "GTCA"]
#     >>> one_hot = convert_one_hot(seqs)
#     >>> one_hot
#     array([[[1., 0., 0., 0.],
#             [0., 1., 0., 0.],
#             [0., 0., 1., 0.],
#             [0., 0., 0., 1.]],

#            [[0., 0., 1., 0.],
#             [0., 0., 0., 1.],
#             [0., 1., 0., 0.],
#             [1., 0., 0., 0.]]])
#     >>> one_hot_shuffled = shuffle_onehot(one_hot)
#     >>> one_hot_shuffled
#     array([[[0., 0., 0., 1.],
#             [0., 1., 0., 0.],
#             [1., 0., 0., 0.],
#             [0., 0., 1., 0.]],

#            [[1., 0., 0., 0.],
#             [0., 0., 1., 0.],
#             [0., 1., 0., 0.],
#             [0., 0., 0., 1.]]])
#     """
    
#     if k == 1:
#         L = one_hot.shape[1] # one_hot has shape (N, L, A)
#         rng = np.random.default_rng()
#         one_hot_shuffled = []

#         for x in one_hot:
#             perm = rng.permutation(L)
#             x_shuffled = x[perm, :]
#             one_hot_shuffled.append(x_shuffled)

#         one_hot_shuffled = np.array(one_hot_shuffled)

#         return one_hot_shuffled
    
#     elif k >= 2:
#         seqs = [seq.str.cat() for seq in convert_onehot_to_sequence(one_hot)] # convert one_hot to pandas Series of letters, then string letters together (for each Series)
#         # seqs = convert_seq(one_hot)
#         seqs_shuffled = []
    
#         for i, seq in enumerate(seqs):
#             seq = seq.upper()
#             seq_shuffled = shuffle( bytes(seq, "utf-8") , k).decode("utf-8") # dinucleotide shuffle

#             seqs_shuffled.append(seq_shuffled)
        
#         one_hot_shuffled = convert_one_hot(seqs_shuffled)
#         return one_hot_shuffled
    
#     else:
#         raise ValueError("k must be an integer greater than or equal to 1")

        
# def shuffle_sequences(sequences, k=1):
#     """Shuffle sequences while preserving k-let frequencies.
    
#     Parameters
#     ----------
#     sequences : array (list, np.ndarray) of strings
#         array of strings; should be one-dimensional
#     k : int
#         k of k-let frequencies to preserve (e.g., with k = 2, dinucleotide
#         shuffle is performed); default is k = 1 (i.e., single-nucleotide
#         shuffle)

#     Returns
#     -------
#     Numpy array of one-hot represented shuffled sequences, of the same shape
#     as one_hot.

#     Examples
#     --------
#     >>> seqs = ["AGCGTTCAA", "TACGAATCG"]
#     >>> seqs_shuffled = shuffle_sequences(seqs, k=2) # dinucleotide shuffle
#     >>> seqs_shuffled
#     ['AAGTTCGCA', 'TCGATAACG']
#     """
#     sequences_shuffled = []

#     for i, seq in enumerate(sequences):
#         seq = seq.upper()
#         seq_shuffled = shuffle( bytes(seq, "utf-8") , k).decode("utf-8")

#         sequences_shuffled.append(seq_shuffled)

#     return sequences_shuffled



def conv_simil_dist(conv_kernel, stride=1):
    [c_out, c_in, kernel_size] = conv_kernel.shape
    padding = ((kernel_size - 1) // stride) * stride
    conv_kernel_l2normalized = torch.nn.functional.normalize(conv_kernel, p=2, dim=(1,2))
    simil = torch.conv1d(conv_kernel_l2normalized, conv_kernel_l2normalized, stride=stride, padding=padding)
    simil_maxpooled = torch.squeeze( torch.nn.AdaptiveMaxPool1d(1)(simil), dim=-1 )
    simil_maxpooled.fill_diagonal_(0)
    return torch.linalg.norm(simil_maxpooled)


def conv_orth_dist(conv_kernel, stride=1):
    [c_out, c_in, kernel_size] = conv_kernel.shape
    P = ((kernel_size - 1) // stride) * stride
    output = torch.conv1d(conv_kernel, conv_kernel, stride=stride, padding=P)
    target = torch.zeros((c_out, c_out, output.shape[-1])).to(output.device)
    ct = int(np.floor(output.shape[-1]/2))
    target[:,:,ct] = torch.eye(c_out)
    return torch.linalg.norm(output - target)


def init_ConvolutionOrthogonal1D(shape, gain=1.0):
    """Initializer that generates a 1D orthogonal kernel for ConvNets.

    The shape of the tensor must have length 3. The number of input
    filters must not exceed the number of output filters.
    The orthogonality (isometry) is exact when the inputs are circular padded.
    There are finite-width effects with non-circular padding (e.g. zero padding).
    See algorithm 1 in (Xiao et al., 2018).

    Args:
      gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
        The 2-norm of an input is multiplied by a factor of `gain` after applying
        this convolution.
    References:
        [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
        ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
    """
    if len(shape) != 3:
        raise ValueError("The tensor to initialize must be three-dimensional")

    c_out, c_in, kernel_size = shape # with TensorFlow: kernel_size, c_in, c_out = shape

    if c_in > c_out:
        raise ValueError("In_filters cannot be greater than out_filters.")

    kernel = _orthogonal_kernel(kernel_size, c_in, c_out)
    kernel *= gain
    return torch.transpose(kernel, 0, 2) # swap c_out and kernel_size dimensions to align with Pytorch

def _orthogonal_matrix(n):
    """Construct an n x n orthogonal matrix.

    Args:
      n: Dimension.

    Returns:
      A n x n orthogonal matrix.
    """
    return torch.nn.init.orthogonal_( torch.empty(n,n) )

def _symmetric_projection(n):
    """Compute a n x n symmetric projection matrix.

    Args:
      n: Dimension.

    Returns:
      A n x n symmetric projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
    """
    q = _orthogonal_matrix(n)
    mask = torch.normal(mean=0, std=1, size=(n,)) > 0 # used to randomly zero out some columns of q
    c = torch.mul(q, mask) # effectively a projection of mask onto space spanned by q
    return torch.matmul(c, c.T) # symmetrizes matrix

def _dict_to_tensor(x, k):
    """Convert a dictionary to a tensor.

    Args:
      x: A dictionary of length k.
      k: Dimension of x.

    Returns:
      A tensor with the same dimension.
    """
    return torch.stack([x[i] for i in range(k)])

def _block_orth(projection_matrix):
    """Construct a kernel.

    Used to construct orthgonal kernel.

    Args:
      projection_matrix: A symmetric projection matrix of size n x n.

    Returns:
      [projection_matrix, (1 - projection_matrix)].
    """
    n = projection_matrix.shape[0]
    kernel = {}
    eye = torch.eye(n)
    kernel[0] = projection_matrix
    kernel[1] = eye - projection_matrix
    return kernel

def _matrix_conv(m1, m2):
    """Matrix convolution.

    Args:
      m1: A dictionary of length k, each element is a n x n matrix.
      m2: A dictionary of length l, each element is a n x n matrix.

    Returns:
      (k + l - 1) dictionary, where each element is a n x n matrix.
    Raises:
      ValueError: If the entries of m1 and m2 are of different dimensions.
    """

    n = (m1[0]).shape[0]
    if n != (m2[0]).shape[0]:
        raise ValueError("The entries in matrices m1 and m2 must have the same dimensions")
    k = len(m1)
    l = len(m2)
    result = {}
    size = k + l - 1
    # Compute matrix convolution between m1 and m2.
    for i in range(size):
        result[i] = torch.zeros(n, n)
        for index in range(min(k, i + 1)):
            if (i - index) < l:
                result[i] += torch.matmul(m1[index], m2[i - index])
    return result

def _orthogonal_kernel(ksize, cin, cout):
    """Construct orthogonal kernel for convolution.

    Args:
      ksize: Kernel size.
      cin: Number of input channels.
      cout: Number of output channels.

    Returns:
      An [ksize, cin, cout] orthogonal kernel.
    Raises:
      ValueError: If cin > cout.
    """
    if cin > cout:
        raise ValueError("The number of input channels cannot exceed the number of output channels.")
    orth = _orthogonal_matrix(cout)[0:cin, :]
    if ksize == 1:
        return torch.unsqueeze(orth, 0)
    p = _block_orth(_symmetric_projection(cout))
    for _ in range(ksize - 2):
        temp = _block_orth(_symmetric_projection(cout))
        p = _matrix_conv(p, temp)
    for i in range(ksize):
        p[i] = torch.matmul(orth, p[i])

    return _dict_to_tensor(p, ksize)


def abbreviate_augmentation_string(augmentation_string):
    if augmentation_string != "":
        aug_string = augmentation_string.lower()
        augs = aug_string.split(",")
        set_augs = ["invert_seqs", "invert_rc_seqs", "delete_seqs", "roll", "roll_seqs", "insert", "insert_seqs", "rc", "mutate_seqs", "noise_gauss"]

        assert all(aug in set_augs for aug in augs), "unrecognized augmentation in augmentation_string"
        assert not ("invert_seqs" in augs and "invert_rc_seqs" in augs), "cannot have both \"invert_seqs\" and \"invert_rc_seqs\" in augmentation_string"
        assert not ("insert" in augs and "insert_seqs" in augs), "cannot have both \"insert\" and \"insert_seqs\" in augmentation_string"
        assert not ("roll" in augs and "roll_seqs" in augs), "cannot have both \"roll\" and \"roll_seqs\" in augmentation_string"


        augs_order = {"invert_seqs" : 6, # invert first to ensure non-random, meaningful sequence is inverted
                      "invert_rc_seqs" : 6,
                      "delete_seqs" : 5, # delete second to also ensure meaningful sequence deleted
                      "roll" : 4,
                      "roll_seqs" : 4, 
                      "insert" : 3, # do insertion after roll to preserve flanking random sequences
                      "insert_seqs" : 3, 
                      "rc" : 2, 
                      "mutate_seqs" : 1,
                      "noise_gauss" : 0 } # add Gaussian noise last to "mutate" all one-hot positions
        augs_sorted = sorted(augs, key=lambda d: augs_order[d])
        
        augs_to_abbrev = {"invert_seqs" : "V",
                          "invert_rc_seqs" : "Vrc", 
                          "delete_seqs" : "D", 
                          "roll" : "L", "roll_seqs" : "L", 
                          "insert" : "I", "insert_seqs" : "I", 
                          "rc" : "R", 
                          "mutate_seqs" : "M",
                          "noise_gauss" : "G"}

        return "".join([augs_to_abbrev[aug] for aug in augs_sorted])
    
    else:
        return ""


# Classes

class TensorDataset(torch.utils.data.Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (torch.Tensor): torch tensors.
    """

    def __init__(self, *tensors: torch.Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)




# Functions

def visualize_first_layer_filters(encoder, x_test, threshold=0.5, window=20, device='cpu', outfile=None):
    """visualize_first_layer_filters: visualize first layer convolutional filters from a
        GRIM model encoder
    
    Arguments:
        encoder: GRIM model encoder (not full GRIM model), which has:
            - encoder.conv1: layer, torch.nn.Conv1d(4, d, m, padding=(m//2), bias=False)
            - encoder.batchnorm1: layer, torch.nn.BatchNorm1d(d)
            - encoder.activation: layer, torch.nn.ReLU()
        x_test: torch tensor of test sequences with which to visualize first layer filters
        threshold: threshold proportion of maximum per-filter activation value
            for including regions in alignment generation (default: 0.5)
        window: window size of alignment, in nucleotides (default: 20)
        device: CUDA (or CPU) device on which to place model and x_test (default: 'cpu')
        outfile: desired output file to which to save visualized first-layer filters;
            if None, file will not be saved (default: None)
    """
    # Set up some variables
    num_filters = encoder.conv1.weight.shape[0] # d, or out_channels value from first convolutional layer
    N_test, A, L = x_test.shape # N = number of test seqs, A = alphabet size (number of nucl.), L = length of seqs

    # Get feature maps of first convolutional layer after activation
    encoder = encoder.to(device)
    encoder = encoder.eval()
    _, fmap = encoder( x_test.to(device) )
    fmap = fmap.detach().cpu().numpy().transpose([0,2,1]) # NB: transpose along axes 1 and 2 to align with TensorFlow implementation

    # Set the left and right window sizes
    window_left = int(window/2)
    window_right = window - window_left
    
    x_test_transpose = x_test.cpu().numpy().transpose([0,2,1]) # NB: transpose along axes 1 and 2 to align with TensorFlow implementation

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
                seq = x_test_transpose[data_index[i], start_window:end_window, :] # NB: in the TF implementation, x_test (not x_test_transpose) is used here
                seq_align.append(seq)

        # Calculate position probability matrix
        if len(seq_align) > 0:
            W.append(np.mean(seq_align, axis=0))
        else:
            W.append(np.zeros((window, A)))
    W = np.array(W)

    # Visualize first-layer filters
    fig = plt.figure(figsize=(30.0, 5*(num_filters/32)))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    num_cols = 8
    num_widths = int(np.ceil(num_filters/num_cols))
    for n, w in enumerate(W):
        ax = fig.add_subplot(num_widths, num_cols, n+1)

        # Calculate sequence logo heights -- information
        I = np.log2(4) + np.sum(w * np.log2(w+1e-7), axis=1, keepdims=True)
        logo = I*w

        # Create DataFrame for logomaker
        filter_len = w.shape[0]
        counts_df = pd.DataFrame(data=0.0, columns=list('ACGT'), index=list(range(filter_len)))
        for a in range(A):
            for l in range(filter_len):
                counts_df.iloc[l,a] = logo[l,a]

        # Plot filter representation
        logomaker.Logo(counts_df, ax=ax)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(0, 2) # set y-axis of all sequence logos to run from 0 to 2 bits
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])

    # If desired, save visualized filters to file
    if outfile is not None:
        fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
        print("Saved visualized convolutional filter alignments to file ", outfile)


class LinearConv(torch.nn.Module):
    def __init__(self, channels, filters, kernel_size, padding=0, stride=1, groups=1, bias=False):
        super(LinearConv, self).__init__()
        self.filters = filters
        self.times = 2 # ratio 1/2
        self.kernel_size = kernel_size
        self.channels = channels//groups
        self.padding = padding
        self.stride = (stride, ) # for consistency with torch.nn.conv1d
        self.stride_int = stride
        self.biasTrue = bias
        self.groups = groups

        self.weight = nn.Parameter(torch.Tensor(filters // self.times, channels, kernel_size))
        self.linear_weights = nn.Parameter(torch.Tensor(filters - filters // self.times, filters // self.times))
        
        torch.nn.init.xavier_uniform_(self.weight)
        self.linear_weights.data.uniform_(-0.1, 0.1)
        
        if self.biasTrue:
            self.bias = nn.Parameter(torch.Tensor(filters))
            self.bias.data.uniform_(-0.1, 0.1)
    
    def forward(self, x):
        correlated_weights = torch.mm(self.linear_weights, self.weight.reshape(self.filters // self.times,-1)).reshape(self.filters - self.filters // self.times, self.channels, self.kernel_size)
        
        if self.biasTrue:
            return F.conv1d(x, torch.cat((self.weight, correlated_weights), dim = 0),
                            bias=self.bias, padding=self.padding, stride=self.stride_int)
        else:
            return F.conv1d(x, torch.cat((self.weight, correlated_weights), dim = 0),
                            padding=self.padding, stride=self.stride_int)