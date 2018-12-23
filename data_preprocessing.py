from definitions import *
import cv2
import numpy as np
from tqdm import tqdm
from keras.utils import to_categorical


def read_csv(file_name):
    new_data = []

    with open(file_name, 'r') as file:
        data = file.readlines()
        for line in data:
            new_line = line.replace('\n', '')
            new_line = new_line.split(',')
            new_data.append(new_line)
    return new_data


def get_image(id, color):
    image_name = id + '_' + color + '.png'
    image_path = path_join(train_dir, image_name)

    image = cv2.imread(image_path, 0)
    resized_image = cv2.resize(image, (128, 128))

    return resized_image


def normalize_image(image):
    return image / 255


def read_images_with_id(id):
    colors = ['blue', 'green', 'red', 'yellow']

    image = []
    for color in colors:
        resized_image = get_image(id, color)
        image.append(resized_image)

    image = np.array(image)
    normalized_image = normalize_image(image)

    return normalized_image


def save_images(images, ids, batch_name):
    grouped_image_file_name = path_join(train_images_dir, batch_name)
    np.savez_compressed(grouped_image_file_name, images=images, ids=ids)


def convert_images_to_npz(batch_size=1024):
    # output_dir
    data = read_csv(train_file)
    data = data[1:]

    images = []
    ids = []
    current_batch_size = 0
    batches_count = 0
    get_batch_name = lambda number: 'batch_' + str(number) + '.npz'

    for line in tqdm(data):

        current_batch_size += 1
        if current_batch_size > batch_size:

            batches_count += 1
            batch_name = get_batch_name(batches_count)
            images = np.array(images)
            ids = np.array(ids)

            save_images(images, ids, batch_name)

            images = []
            ids = []
            current_batch_size = 0

        id = line[0]
        normalized_image = read_images_with_id(id)

        ids.append(id)
        images.append(normalized_image)

    batches_count += 1
    batch_name = get_batch_name(batches_count)
    images = np.array(images)
    ids = np.array(ids)
    save_images(images, ids, batch_name)


def str_to_categorical(target):
    answer = np.zeros(28)
    for target_part in target.split(' '):
        categorical_target_part = to_categorical(target_part, 28)
        answer += categorical_target_part
    return answer


def convert_target_to_npz(batch_size=1024):
    data = read_csv(train_file)

    categorical_data = []
    ids = []
    for id, target in data[1:]:
        categorical_target = str_to_categorical(target)
        categorical_data.append(categorical_target)
        ids.append(id)
    categorical_data = np.array(categorical_data)
    ids = np.array(ids)

    start_elements = np.array(list(range(0, categorical_data.shape[0], batch_size)))
    end_elements = start_elements + batch_size
    end_elements[-1] = categorical_data.shape[0]

    for i, (start, end) in enumerate(zip(start_elements, end_elements)):
        batch_name = 'batch_{}.npz'.format(i)
        grouped_image_file_name = path_join(train_target_dir, batch_name)

        batch_data = categorical_data[start:end]
        batch_ids = ids[start:end]
        np.savez_compressed(grouped_image_file_name, targets=batch_data, ids=batch_ids)


def main():
    # convert_images_to_npz()
    # input_str = '1 2 3'
    # answer = str_to_categorical(input_str)
    # print(answer)
    convert_target_to_npz()
    pass


if __name__ == '__main__':
    main()
    pass
