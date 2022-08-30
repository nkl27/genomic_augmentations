import sys

augmentation_string = sys.argv[1]

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

    print("".join([augs_to_abbrev[aug] for aug in augs_sorted]))

else:
    print("N")
