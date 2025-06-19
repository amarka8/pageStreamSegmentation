import os
import math
import numpy as np
import PyPDF2



def build_val(used_set):
    # use os module to get list of pdfs to train and test on
    dir_path = "data/raw"
    file_lst = os.listdir(dir_path)
    file_lst = [dir_path + "/" + file_name for file_name in file_lst]
    val_len = math.floor(len(file_lst) * 0.125)
    #build train test and val sets through random sampling
    used_set = set()
    val = []

    while len(val) < val_len:
        idxs = np.random.choice(len(file_lst), val_len - len(val))
        for idx in idxs:
            if idx not in used_set:
                val.append(file_lst[idx])
                used_set.add(idx)

    
    # DEBUG
    # print(val)
    # print("\n")
    val_stats = []

    for file_name in val:

        # DEBUG
        print(file_name)
        print("\n")

        write_file = open("val.txt", "a")

        try:
            pdfReader = PyPDF2.PdfReader(open(file_name, "rb"))
        except:
            continue

        # count number of pages
        totalPages = len(pdfReader.pages)
        val_stats.append(totalPages)
        for page in range(totalPages):
            write_file.write(str(page)+"        "+file_name+"\n")

    return val_stats

def build_test(used_set):
    # use os module to get list of pdfs to train and test on
    dir_path = "data/raw"
    file_lst = os.listdir(dir_path)
    file_lst = [dir_path + "/" + file_name for file_name in file_lst]
    test_len = len(file_lst) - len(used_set)
    #build train test and val sets through random sampling
    used_set = set()
    test = []

    while len(test) < test_len:
        idxs = np.random.choice(len(file_lst), test_len - len(test))
        for idx in idxs:
            if idx not in used_set:
                test.append(file_lst[idx])
                used_set.add(idx)

    
    # DEBUG
    # print(test)
    # print("\n")
    test_stats = []

    for file_name in test:

        # DEBUG
        print(file_name)
        print("\n")

        write_file = open("test.txt", "a")

        try:
            pdfReader = PyPDF2.PdfReader(open(file_name, "rb"))
            # count number of pages

        except:
            continue

        totalPages = len(pdfReader.pages)
        test_stats.append(totalPages)
        for page in range(totalPages):
            write_file.write(str(page)+"        "+file_name+"\n")

    return test_stats

def build_train(used_set):
    # use os module to get list of pdfs to train and test on
    dir_path = "data/raw"
    file_lst = os.listdir(dir_path)
    file_lst = [dir_path + "/" + file_name for file_name in file_lst]
    train_len = math.floor(len(file_lst) * 0.75)
    train = []

    while len(train) < train_len:
        idxs = np.random.choice(len(file_lst), train_len - len(train))
        for idx in idxs:
            if idx not in used_set:
                train.append(file_lst[idx])
                used_set.add(idx)

    
    # DEBUG
    # print(train)
    # print("\n")
    train_stats = []

    for file_name in train:

        # DEBUG
        print(file_name)
        print("\n")

        write_file = open("train.txt", "a")

        try:
            pdfReader = PyPDF2.PdfReader(open(file_name, "rb"))

        except:
            continue

        # count number of pages
        totalPages = len(pdfReader.pages)
        train_stats.append(totalPages)

        for page in range(totalPages):
            write_file.write(str(page)+"        "+file_name+"\n")

    return train_stats


def main():
    used_set = set()
    train_pages = build_train(used_set)
    val_pages = build_val(used_set)
    test_pages = build_test(used_set)
    print("Train Statistics")
    print("--------------------------")
    print("Standard Deviation: ")
    print(np.std(train_pages))
    print("Mean: ")
    print(np.mean(train_pages))

    print("Val Statistics")
    print("--------------------------")
    print("Standard Deviation: ")
    print(np.std(val_pages))
    print("Mean: ")
    print(np.mean(val_pages))

    print("Test Statistics")
    print("--------------------------")
    print("Standard Deviation: ")
    print(np.std(test_pages))
    print("Mean: ")
    print(np.mean(test_pages))



if __name__ == "__main__":
    main()