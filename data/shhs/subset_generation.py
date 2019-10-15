import glob
import shutil
import numpy as np
import os
from tqdm import tqdm

np.random.seed(1)
PERCENT = 0.02

def main():


    data_root_path = '/project/mgh/SHHS/'
    data_root_subset_path = '/project/mgh/Rahul_shhs_subset/'

    shhs_edf_path = data_root_path + 'shhs/polysomnography/edfs/shhs1/'
    shhs_edf_subset_path = data_root_subset_path + 'edfs/shhs1/'

    shhs_nsrr_path = data_root_path + 'shhs/polysomnography/annotations-events-nsrr/shhs1/'
    shhs_nsrr_subset_path = data_root_subset_path + 'annotations-events-nsrr/shhs1/'

    edf_files, nsrr_files = glob.glob(shhs_edf_path + '*.edf'), []
    edf_files = list(np.random.choice(edf_files, size=int(len(edf_files)*PERCENT), replace=False))

    for edf_file in edf_files:
        pat_id = edf_file.split('shhs1-')[1].split('.edf')[0]
        nsrr_file = shhs_nsrr_path + 'shhs1-{}-nsrr.xml'.format(pat_id)

        if not os.path.exists(nsrr_file):
            raise Exception('NSRR file not found')

        nsrr_files.append(nsrr_file)

    for (edf_file,nsrr_file) in tqdm(zip(edf_files,nsrr_files)):
        shutil.copy(edf_file,shhs_edf_subset_path)
        shutil.copy(nsrr_file,shhs_nsrr_subset_path)

    # print(len(nsrr_files),len(edf_files))





if __name__ == '__main__':
    main()
