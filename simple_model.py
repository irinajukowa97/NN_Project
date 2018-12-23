from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

from data_loader import Loader


loader = Loader()
model = MLPClassifier()

for batch in loader.get_batches(1):
    images, targets, ids = batch
    processed = np.reshape(images, (images.shape[0], images.shape[1]*images.shape[2]*images.shape[3]))
    model.fit(processed, targets)

for batch in loader.get_batches(1):
    images, targets, ids = batch
    processed = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]))
    score = model.score(processed, targets)
    # score = model.score(targets, prediction)
    print(score)