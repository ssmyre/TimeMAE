import numpy as np
import torch
from scipy.io import arff
import pandas as pd

def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0
    return data


def load_UCR(Path='data/', folder='Cricket'):
    train_path = Path + folder + '/' + folder + '_TRAIN.arff'
    test_path = Path + folder + '/' + folder + '_TEST.arff'
    TRAIN_DATA = []
    TRAIN_LABEL = []
    label_dict = {}
    label_index = 0
    with open(train_path, encoding='UTF-8', errors='ignore') as f:
        data, meta = arff.loadarff(f)
        f.close()
    if type(data[0][0]) == np.ndarray:  # multivariate
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            raw_data_list = raw_data.tolist()
            # print(raw_data_list)
            TRAIN_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            TEST_LABEL.append(label_dict[raw_label])
            raw_data_list = raw_data.tolist()
            TEST_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        index = np.arange(0, len(TRAIN_DATA))
        np.random.shuffle(index)

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA)[index], np.array(TRAIN_LABEL)[index]], \
               [np.array(TRAIN_DATA)[index], np.array(TRAIN_LABEL)[index]], \
               [np.array(TEST_DATA), np.array(TEST_LABEL)]

    else:  # univariate
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            TRAIN_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            TEST_LABEL.append(label_dict[raw_label])
            TEST_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
            np.array(TEST_DATA), np.array(TEST_LABEL)]


def load_HAR(Path='data/HAR/'):
    train = torch.load(Path + 'train.pt')
    val = torch.load(Path + 'val.pt')
    test = torch.load(Path + 'test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = torch.cat([TRAIN_DATA, VAL_DATA])
    ALL_TRAIN_LABEL = torch.cat([TRAIN_LABEL, VAL_LABEL])
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]


def load_mat(Path='data/AUSLAN/'):
    if 'UWave' in Path:
        train = torch.load(Path + 'train_new.pt')
        test = torch.load(Path + 'test_new.pt')
    else:
        train = torch.load(Path + 'train.pt')
        test = torch.load(Path + 'test.pt')
    TRAIN_DATA = train['samples'].float()
    TRAIN_LABEL = (train['labels'] - 1).long()
    TEST_DATA = test['samples'].float()
    TEST_LABEL = (test['labels'] - 1).long()
    print('data loaded')

    return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]

def load_txt(Path='data/', folder = 'FSCV'):
    print('loading_data')
    train_path = Path + folder + '/' + folder + '_TRAIN.txt'
    test_path = Path + folder + '/' + folder + '_TEST.txt'
    TRAIN_DATA = []
    TRAIN_LABEL = []
    label_dict = {}
    label_index = 0

    data = pd.read_csv(train_path, sep='\t')
    for index in range(data.shape[0]):
        raw_data = np.array(list(data.iloc[index,:-1]))
        raw_label = data.iloc[index,-1]
        if label_dict.__contains__(raw_label):
            TRAIN_LABEL.append(label_dict[raw_label])
        else:
            label_dict[raw_label] = label_index
            TRAIN_LABEL.append(label_index)
            label_index += 1
        TRAIN_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))
        
    data = pd.read_csv(test_path, sep='\t')
    TEST_DATA = []
    TEST_LABEL = []
    for index in range(data.shape[0]):
        raw_data = np.array(list(data.iloc[index,:-1]))
        raw_label = data.iloc[index,-1]
        if label_dict.__contains__(raw_label):
            TEST_LABEL.append(label_dict[raw_label])
        else:
            label_dict[raw_label] = label_index
            TEST_LABEL.append(label_index)
            label_index += 1
        TEST_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

    TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
    TEST_DATA = padding_varying_length(np.array(TEST_DATA))

    return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)], label_dict 

def load_txt_regress(Path='data/', folder='FSCV'):
    print('loading_data')
    train_path = f"{Path}{folder}/{folder}_TRAIN.txt"
    test_path = f"{Path}{folder}/{folder}_TEST.txt"

    def load_file(path):
        data = pd.read_csv(path, sep='\t')
        raw_inputs = data.iloc[:, :-3].values.astype(np.float32)  # time series
        raw_labels = data.iloc[:, -2:].values.astype(np.float32)  # dopamine + serotonin
        raw_electrodes = data.iloc[:, -3] # dopamine + serotonin
        reshaped_inputs = [x.reshape(-1, 1) for x in raw_inputs]  # each: [T, 1]
        return reshaped_inputs, raw_labels, raw_electrodes

    TRAIN_DATA, TRAIN_LABELS, TRAIN_ELECTRODES = load_file(train_path)
    TEST_DATA, TEST_LABELS, TEST_ELECTRODES = load_file(test_path)

    TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
    TEST_DATA = padding_varying_length(np.array(TEST_DATA))
    return [np.array(TRAIN_DATA), np.array(TRAIN_LABELS), np.array(TRAIN_ELECTRODES)], \
           [np.array(TRAIN_DATA), np.array(TRAIN_LABELS), np.array(TRAIN_ELECTRODES)], \
           [np.array(TEST_DATA), np.array(TEST_LABELS), np.array(TEST_ELECTRODES)]

def load_txt_valid(Path='data/', folder='FSCV'):
    print('loading_data')
    train_path = f"{Path}{folder}/{folder}_TRAIN.txt"
    valid_path = f"{Path}{folder}/{folder}_VALID.txt"
    test_path = f"{Path}{folder}/{folder}_TEST.txt"

    def load_file(path):
        data = pd.read_csv(path, sep='\t')
        raw_inputs = data.iloc[:, :-3].values.astype(np.float32)  # time series
        raw_labels = data.iloc[:, -2:].values.astype(np.float32)  # dopamine + serotonin
        raw_electrodes = data.iloc[:, -3] # dopamine + serotonin
        reshaped_inputs = [x.reshape(-1, 1) for x in raw_inputs]  # each: [T, 1]
        return reshaped_inputs, raw_labels, raw_electrodes

    TRAIN_DATA, TRAIN_LABELS, TRAIN_ELECTRODES = load_file(train_path)
    VALID_DATA, VALID_LABELS, VALID_ELECTRODES = load_file(valid_path)
    TEST_DATA, TEST_LABELS, TEST_ELECTRODES = load_file(test_path)

    TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
    VALID_DATA = padding_varying_length(np.array(VALID_DATA))
    TEST_DATA = padding_varying_length(np.array(TEST_DATA))
    return [np.array(TRAIN_DATA), np.array(TRAIN_LABELS), np.array(TRAIN_ELECTRODES)], \
           [np.array(VALID_DATA), np.array(VALID_LABELS), np.array(VALID_ELECTRODES)], \
           [np.array(TEST_DATA), np.array(TEST_LABELS), np.array(TEST_ELECTRODES)]

def load_txt_all(Path='data/', folder='FSCV'):
    print('loading_data')
    train_all_path = f"{Path}{folder}/{folder}_TRAIN_ALL.txt"
    train_path = f"{Path}{folder}/{folder}_TRAIN.txt"
    test_path = f"{Path}{folder}/{folder}_TEST.txt"

    def load_file(path):
        data = pd.read_csv(path, sep='\t')
        raw_inputs = data.iloc[:, :-3].values.astype(np.float32)  # time series
        raw_labels = data.iloc[:, -2:].values.astype(np.float32)  # dopamine + serotonin
        raw_electrodes = data.iloc[:, -3] # dopamine + serotonin
        reshaped_inputs = [x.reshape(-1, 1) for x in raw_inputs]  # each: [T, 1]
        return reshaped_inputs, raw_labels, raw_electrodes

    TRAIN_DATA_ALL, TRAIN_LABELS_ALL, TRAIN_ELECTRODES_ALL = load_file(train_all_path)
    TRAIN_DATA, TRAIN_LABELS, TRAIN_ELECTRODES = load_file(train_path)
    TEST_DATA, TEST_LABELS, TEST_ELECTRODES = load_file(test_path)

    TRAIN_DATA_ALL = padding_varying_length(np.array(TRAIN_DATA_ALL))
    TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
    TEST_DATA = padding_varying_length(np.array(TEST_DATA))
    return [np.array(TRAIN_DATA_ALL), np.array(TRAIN_LABELS_ALL), np.array(TRAIN_ELECTRODES_ALL)], \
           [np.array(TRAIN_DATA), np.array(TRAIN_LABELS), np.array(TRAIN_ELECTRODES)], \
           [np.array(TEST_DATA), np.array(TEST_LABELS), np.array(TEST_ELECTRODES)]