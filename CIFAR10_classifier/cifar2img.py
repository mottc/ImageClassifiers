from scipy.misc import imsave
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

for j in range(1, 6):
    fileName = "data_batch_" + str(j)
    trainData = unpickle(fileName)
    print(fileName)
    for i in range(0, 10000):
        img = np.reshape(trainData[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        imgName = 'train/' + str(trainData[b'labels'][i]) + '_' + str(i + (j - 1)*10000) + '.jpg'
        imsave(imgName, img)


testData = unpickle("test_batch")
print("test_batch")
for i in range(0, 10000):
    img = np.reshape(testData[b'data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    imgName = 'test/' + str(testData[b'labels'][i]) + '_' + str(i) + '.jpg'
    imsave(imgName, img)