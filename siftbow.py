import cv2
import numpy as np
import os
import glob
import sys

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 100
COVERCLASS = [0, 1, 3]


def getInputArgs ():
    if len(sys.argv) != 2:
        print (f'\nFormat:\n    {sys.argv[0]}  {"{image path/filename}"}\n')
        exit()

    return sys.argv[1]

def read_images(img_list):
    read = []
    for i in img_list:
        img = cv2.imread(i)
        read.append(img)
    return read

def get_pos_and_neg_paths(i):
    bat_path = 'training_dataset/0/%d.jpeg' % (i+1)
    spider_path = 'training_dataset/2/%d.jpeg' % (i+1)
    neg_path = 'training_dataset/1/%d.jpeg' % (i+1)
    return bat_path, spider_path, neg_path

def add_sample(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        bow_kmeans_trainer.add(descriptors)

def extract_bow_descriptors(img):
    features = sift.detect(img)
    return bow_extractor.compute(img, features)


test_folder = getInputArgs()


batman = glob.glob('training_dataset/0/*.jpeg')
spiderman = glob.glob('training_dataset/2/*.jpeg')
neither = glob.glob('training_dataset/1/*.jpeg')


if test_folder == '1':
    test_dataset = glob.glob('test_dataset/*')
else:
    test_dataset = glob.glob('test_comic_locations/*')

bat_tr = read_images(batman)
spider_tr = read_images(spiderman)

neither_tr = read_images(neither)
test_images = read_images(test_dataset)


sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params, search_params)


bow_kmeans_trainer = cv2.BOWKMeansTrainer(60)
bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)

for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
    bat_path, spider_path, neg_path = get_pos_and_neg_paths(i)
    add_sample(bat_path)
    add_sample(spider_path)
    add_sample(neg_path)

voc = bow_kmeans_trainer.cluster()
bow_extractor.setVocabulary(voc)



training_data = []
training_labels = []
for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):

    bat_path, spider_path, neg_path = get_pos_and_neg_paths(i)

    bat_img = cv2.imread(bat_path, cv2.IMREAD_GRAYSCALE)
    bat_descriptors = extract_bow_descriptors(bat_img)
    if bat_descriptors is not None:
        training_data.extend(bat_descriptors)
        training_labels.append(1)

    spider_img = cv2.imread(spider_path, cv2.IMREAD_GRAYSCALE)
    spider_descriptors = extract_bow_descriptors(spider_img)
    if spider_descriptors is not None:
        training_data.extend(spider_descriptors)
        training_labels.append(2)

    neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
    neg_descriptors = extract_bow_descriptors(neg_img)
    if neg_descriptors is not None:
        training_data.extend(neg_descriptors)
        training_labels.append(-1)

print(len(training_data))
print(len(training_labels))
svm = cv2.ml.SVM_create()
svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
            np.array(training_labels))

for test_img_path in test_dataset:
    img = cv2.imread(test_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptors = extract_bow_descriptors(gray_img)
    prediction = svm.predict(descriptors)
    print(prediction)
    if prediction[1][0][0] == 1.0:
        text = 'batman'
        color = (0, 255, 0)

    elif prediction[1][0][0] == 2.0:
        text = 'spiderman'
        color = (0, 255, 0)
        
    else:
        text = 'neither'
        color = (0, 0, 255)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color, 2, cv2.LINE_AA)
    cv2.imshow(test_img_path, img)

cv2.waitKey(0)

'''
x, y = get_pos_and_neg_paths_batman(4)
x = cv2.imread(x)


cv2.imshow('image', test_images[0])
cv2.imshow('image 2 ', x)
cv2.waitKey(0)
cv2.destroyAllWindows()'''