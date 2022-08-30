import torch
import torch.nn as nn
import torch.nn.functional as F


def aug_invert_seqs(x, invert_min=100, invert_max=200):
    """augment *batch* of sequences x by directly inverting a randomly chosen 
        stretch of nucleotides from each seq. (differently chosen for *each seq.*)
        
        x: batch of sequences (shape: (N_batch, A, L))
        invert_min (optional): min length of inversion (default: 100)
        invert_max (optional): max length of inversion (default: 200)
    """
    N_batch, A, L = x.shape
    inversion_lens = torch.randint(invert_min, invert_max + 1, (N_batch,))
    inversion_inds = torch.randint(L - invert_max + 1, (N_batch,)) # inversion must be in boundaries of seq.
    
    x_aug = []
    for seq, inversion_len, inversion_ind in zip(x, inversion_lens, inversion_inds):
        x_aug.append( torch.cat([seq[:,:inversion_ind], 
                                 torch.flip(seq[:,inversion_ind:inversion_ind+inversion_len], dims=[1]), 
                                 seq[:,inversion_ind+inversion_len:]],
                                -1) )
    return torch.stack(x_aug)


def aug_invert_rc_seqs(x, invert_min=100, invert_max=200):
    """augment *batch* of sequences x by inverting a randomly chosen stretch
        of nucleotides to its reverse complement in each sequence (with each stretch 
        being independently and randomly chosen for *each sequence*)
        
        x: batch of sequences (shape: (N_batch, A, L))
        invert_min (optional): min length of reverse complement inversion (default: 100)
        invert_max (optional): max length of reverse complement inversion (default: 200)
    """
    N_batch, A, L = x.shape
    inversion_lens = torch.randint(invert_min, invert_max + 1, (N_batch,))
    inversion_inds = torch.randint(L - invert_max + 1, (N_batch,)) # inversion must be in boundaries of seq.
        
    x_aug = []
    for seq, inversion_len, inversion_ind in zip(x, inversion_lens, inversion_inds):
        x_aug.append( torch.cat([seq[:,:inversion_ind], 
                                 torch.flip(seq[:,inversion_ind:inversion_ind+inversion_len], dims=[0,1]), 
                                 seq[:,inversion_ind+inversion_len:]],
                                -1) )
    return torch.stack(x_aug)


def aug_delete_seqs(x, delete_min=100, delete_max=200):
    """augment *batch* of sequences x by deleteing a stretch of nucleotides
        of length between delete_min and delete_max randomly chosen
        from each sequence at a *different* randomly chosen position (for *each sequence*);
        to preserve sequence length, a stretch of random nucleotides of length equal to 
        the deletion length is split evenly between the beginning and end of the sequence
        
        x: batch of sequences (shape: (N_batch, A, L))
        delete_min (optional): min length of deletion (default: 100)
        delete_max (optional): max length of deletion (default: 200)
    """
    N_batch, A, L = x.shape
    a = torch.eye(A)
    p = torch.tensor([1/A for _ in range(A)])
    insertions = torch.stack([a[p.multinomial(delete_max, replacement=True)].transpose(0,1) for _ in range(N_batch)]).to(x.device)
    
    delete_lens = torch.randint(delete_min, delete_max + 1, (N_batch,))
    delete_inds = torch.randint(L - delete_max + 1, (N_batch,)) # deletion must be in boundaries of seq.
    
    x_aug = []
    for seq, insertion, delete_len, delete_ind in zip(x, insertions, delete_lens, delete_inds):
        insert_beginning_len = torch.div(delete_len, 2, rounding_mode='floor').item()
        insert_end_len = delete_len - insert_beginning_len
        
        x_aug.append( torch.cat([insertion[:,:insert_beginning_len],
                                 seq[:,0:delete_ind], 
                                 seq[:,delete_ind+delete_len:],
                                 insertion[:,delete_max-insert_end_len:]],
                                -1) )
    return torch.stack(x_aug)


def aug_roll(x, shift_min=100, shift_max=200):
    """augment *batch* of sequences x by rolling all sequences along the position dim
        by a different, randomly chosen amount between shift_min and shift_max
        (see <https://pytorch.org/docs/stable/generated/torch.roll.html>)
        
        x: batch of sequences (shape: (N_batch, A, L))
        shift_min (optional): min number of places by which position can be shifted (default: 100)
        shift_max (optional): max number of places by which position can be shifted (default: 200)
    """
    shift = torch.randint(shift_min, shift_max + 1, (1,)).item()
    return torch.roll(x, shift, -1)


def aug_roll_seqs(x, shift_min=100, shift_max=200):
    """augment *batch* of sequences x by rolling *each sequence* along the position dim
        by a different, randomly chosen amount between shift_min and shift_max
        (see <https://pytorch.org/docs/stable/generated/torch.roll.html>)
        
        x: batch of sequences (shape: (N_batch, A, L))
        shift_min (optional): min number of places by which position can be shifted (default: 100)
        shift_max (optional): max number of places by which position can be shifted (default: 200)
    """
    N_batch = x.shape[0]
    shifts = torch.randint(shift_min, shift_max + 1, (N_batch,))
    ind_neg = torch.rand(N_batch) < 0.5
    shifts[ind_neg] = -1 * shifts[ind_neg]
    x_rolled = []
    for i, shift in enumerate(shifts):
        x_rolled.append( torch.roll(x[i], shift.item(), -1) )
    x_rolled = torch.stack(x_rolled).to(x.device)
    return x_rolled


def aug_insert(x, insert_min=100, insert_max=200):
    """augment *batch* of sequences x by inserting a stretch of random nucleotides
        into each sequence at the same randomly chosen position (for each batch);
        remaining nucleotides up to insert_max are split between beginning and end
        of sequences
        
        x: batch of sequences (shape: (N_batch, A, L))
        insert_min (optional): min length of insertion (default: 100)
        insert_max (optional): max length of insertion (default: 200)
    """
    insert_len = torch.randint(insert_min, insert_max + 1, (1,)).item() 
    insert_beginning_len = torch.div((insert_max - insert_len), 2, rounding_mode='floor').item()
    insert_end_len = insert_max - insert_len - insert_beginning_len

    N_batch, A, L = x.shape
    a = torch.eye(A)
    p = torch.tensor([1/A for _ in range(A)])
    insertion = torch.stack([a[p.multinomial(insert_max, replacement=True)].transpose(0,1) for _ in range(N_batch)]).to(x.device)

    insert_ind = torch.randint(L, (1,)).item()
    x_aug = torch.cat([insertion[:,:,:insert_beginning_len],
                       x[:,:,:insert_ind], 
                       insertion[:,:,insert_beginning_len:insert_beginning_len+insert_len], 
                       x[:,:,insert_ind:],
                       insertion[:,:,insert_beginning_len+insert_len:insert_max]],
                      -1)
    return x_aug


def aug_insert_seqs(x, insert_min=100, insert_max=200):
    """augment *batch* of sequences x by inserting a stretch of random nucleotides
        into each sequence at a *different* randomly chosen position (for *each sequence*);
        remaining nucleotides up to insert_max are split between beginning and end
        of sequences
        
        x: batch of sequences (shape: (N_batch, A, L))
        insert_min (optional): min length of insertion (default: 100)
        insert_max (optional): max length of insertion (default: 200)
    """
    N_batch, A, L = x.shape
    a = torch.eye(A)
    p = torch.tensor([1/A for _ in range(A)])
    insertions = torch.stack([a[p.multinomial(insert_max, replacement=True)].transpose(0,1) for _ in range(N_batch)]).to(x.device)

    insert_lens = torch.randint(insert_min, insert_max + 1, (N_batch,))
    insert_inds = torch.randint(L, (N_batch,))

    x_aug = []
    for seq, insertion, insert_len, insert_ind in zip(x, insertions, insert_lens, insert_inds):
        insert_beginning_len = torch.div((insert_max - insert_len), 2, rounding_mode='floor').item()
        insert_end_len = insert_max - insert_len - insert_beginning_len
        x_aug.append( torch.cat([insertion[:,:insert_beginning_len],
                                 seq[:,:insert_ind], 
                                 insertion[:,insert_beginning_len:insert_beginning_len+insert_len], 
                                 seq[:,insert_ind:],
                                 insertion[:,insert_beginning_len+insert_len:insert_max]],
                                -1) )
    return torch.stack(x_aug)


def aug_rc(x, rc_prob=1.0):
    """augment *batch* of sequences x by returning the batch with each sequence 
        either kept as the original sequence or \"mutated\" to its reverse complement
        with probability rc_prob 
        
        x: batch of sequences (shape: (N_batch, A, L))
        rc_prob (optional): probability of each sequence to be \"mutated\" to its 
            reverse complement (default: 1.0)
    """
    x_aug = torch.clone(x)
    ind_rc = torch.rand(x_aug.shape[0]) < rc_prob
    x_aug[ind_rc] = torch.flip(x_aug[ind_rc], dims=[1,2])
    return x_aug


def aug_mutate_seqs(x, mutate_frac=0.1):
    """augment *batch* of sequences x by randomly mutating a fraction mutate_frac
        of each sequence's nucleotides, randomly chosen independently for each 
        sequence in the batch
        
        x: batch of sequences (shape: (N_batch, A, L))
        mutate_frac (optional): fraction of each sequence's nucleotides to mutate 
            (default: 0.1)
    """
    N_batch, A, L = x.shape
    num_mutations = round(mutate_frac / 0.75 * L) # num. mutations per sequence (accounting for silent mutations)
    mutation_inds = torch.argsort(torch.rand(N_batch,L))[:, :num_mutations] # see <https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146>0

    a = torch.eye(A)
    p = torch.tensor([1/A for _ in range(A)])
    mutations = torch.stack([a[p.multinomial(num_mutations, replacement=True)].transpose(0,1) for _ in range(N_batch)]).to(x.device)
    
    x_aug = torch.clone(x)
    for i in range(N_batch):
        x_aug[i,:,mutation_inds[i]] = mutations[i]
        
    return x_aug


def aug_noise_gauss(x, std=0.1):
    """augment *batch* of sequences x by returning a noise-added versions of x,
        both with independent Gaussian noise (mean 0, standard deviation std) added
        
        x: batch of sequences (shape: (N_batch, A, L))
        std (optional): standard deviation of Gaussian noise added (default: 0.1)
    """
    return x + torch.normal(0, std, x.shape).to(x.device)


def aug_pad_end(x, insert_max=200):
    """augment *batch* of sequences x by inserting a stretch of random nucleotides--that is,
        padding--of fixed length insert_max at the end (as read/inputted) of each sequence;
        this "augmentation" is typically used to simply pad out sequences to the expected 
        input length for models using an insertion augmentation
        
        x: batch of sequences (shape: (N_batch, A, L))
        insert_max (optional): length of insertion, notated with variable insert_max
            to match the aug_insert() and aug_insert_seqs() functions (default: 200)
    """
    N_batch, A, L = x.shape
    a = torch.eye(A)
    p = torch.tensor([1/A for _ in range(A)])
    padding = torch.stack([a[p.multinomial(insert_max, replacement=True)].transpose(0,1) for _ in range(N_batch)]).to(x.device)
    x_padded = torch.cat( [x, padding.to(x.device)], dim=2 )
    
    return x_padded

def aug_crop(x, frac_crop=0.9):
    """augment *batch* of sequences x by taking two possibly overlapping crops:
        one of the first frac_crop of the sequence, one of the latter frac_crop
        
        x: batch of sequences (shape: (N_batch, A, L))
        frac_crop (optional): fraction of sequence to keep in each crop (default: 0.75)
    """
    L = x.shape[2]
    return x[:, :, :int(frac_crop * L)], x[:, :, (L - int(frac_crop * L)):]

