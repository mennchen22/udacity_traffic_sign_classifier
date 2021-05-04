if __name__ == "__main__":
    import numpy as np
    array = np.zeros((32,32))
    print(array.shape)
    array = array.reshape(1,32,32,1)
    print(array.shape)

    from readTrafficSigns import readTrafficSigns
    from matplotlib import pyplot as plt

    trainImages, trainLabels = readTrafficSigns(
        "./traffic-signs-data/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/")
    print(f"Labels {len(trainLabels)} and  Images {len(trainImages)}")
    plt.imshow(trainImages[42])
    plt.show()
