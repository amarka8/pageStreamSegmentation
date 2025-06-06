import os
import math

def main():
    # use os module to get list of pdfs to train and test on
    dir_path = "/Users/amarkanaka/Desktop/server_baker_files_final"
    file_lst = os.listdir(dir_path)
    file_lst = [dir_path + "/" + file_name for file_name in file_lst]
    train_len = math.floor(len(file_lst) * 0.125)
    val_len = train_len
    test_len = len(file_lst) - val_len - train_len
    # create folders to be itemized 


    # names = []
    # for name in glob(f'{path_list}/*'):
    #     name = name.split('/')[-1]
    #     names.append(name)

    # for i, n in enumerate(np.random.poisson(lam, samples)):
    #     for name in np.random.choice(names, n):
    #         print(name, i + 1)


if __name__ == "__main__":
    main()