import numpy as np
from dataset.ei_dataset import EdgeImpulseDataset

if __name__ == '__main__':
    validation_dataset = EdgeImpulseDataset('data/EI_dataset/testing')
    validation_data = []
    validation_labels = []
    for i in range(len(validation_dataset)):
        data, label = validation_dataset[i]
        validation_data.append(data.reshape(1, -1))
        validation_labels.append(label)
    validation_data = np.array(validation_data)
    print(validation_data.shape)
    np.save('validation_data.npy', validation_data)