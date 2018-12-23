import os


def path_join(left, right):
    return os.path.join(left, right)

def create_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

data_dir = 'C:/Users/mstrHW/Desktop/Irina/HumanProteinAtlas_ToSend'
train_file = path_join(data_dir, 'train.csv')
test_file = path_join(data_dir, 'sample_submission.csv')

train_dir = path_join(data_dir, 'train')
test_dir = path_join(data_dir, 'test')

train_images_dir = path_join(data_dir, 'train_images_npz')
create_directory(train_images_dir)

train_target_dir = path_join(data_dir, 'train_target_npz')
create_directory(train_target_dir)
