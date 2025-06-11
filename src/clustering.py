import os
import math
import numpy as np
import PyPDF2

def build_train_val_test():
    # use os module to get list of pdfs to train and test on
    dir_path = "/Users/amarkanaka/Desktop/server_baker_files_final"
    file_lst = os.listdir(dir_path)
    file_lst = [dir_path + "/" + file_name for file_name in file_lst]
    train_len = math.floor(len(file_lst) * 0.125)
    test_len = len(file_lst) - 2 * train_len 
    #build train test and val sets through random sampling
    used_set = set()
    train = []
    val = []
    test = []

    while len(train) < train_len:
        idxs = np.random.choice(len(file_lst), train_len - len(train))
        for idx in idxs:
            if idx not in used_set:
                train.append(file_lst[idx])
                used_set.add(idx)

    
    # DEBUG
    print(train)
    print("\n")


    for file_name in train:

        # DEBUG
        print(file_name)
        print("\n")

        write_file = open("test.txt", "a")

        pdfReader = PyPDF2.PdfReader(open(file_name, "r"))
        # count number of pages
        totalPages = len(pdfReader.pages)

        for page in range(totalPages):
            write_file.write(str(page)+"        "+file_name+"\n")

def main():
    build_train_val_test()



if __name__ == "__main__":
    main()