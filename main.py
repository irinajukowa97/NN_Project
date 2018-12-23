from definitions import *
from data_loader import Loader


loader = Loader()
for batch in loader.get_batches():
    print(batch)