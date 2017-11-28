import os
import pickle

cur_dir = os.path.dirname(__file__)

clf = pickle.load(open(os.path.join(cur_dir, "pkl_objects", "classifier.pkl"), "rb"))