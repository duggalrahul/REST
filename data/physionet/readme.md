# Instructions for preparing the Physionet dataset

1. To generate the bash script for downloading the dataset containing 40 patient's night 1 data run following
	- ```python generate_physionet_bash.py```
	- This creates a new folder ```raw/``` which contains the file ```physionet_bash.sh``` inside it.

2. Enable write priveledges
	- ```chmod +x raw/get_physionet.sh```

3. Run bash script download dataset
	- ```bash raw/get_physionet.sh```

4. Run following scripts to get processed data (**This requires python 2.7 due to third party code!!**) 
	- create a new python 2.7 environment and install required libraries
		- ```conda create -n physionet_data python=2.7```
		- ```conda activate physionet_data```
		- ```pip install mne==0.17.2```
		- ```pip install pathlib```
	- Create data with following command 
		- Extended version : ```python preprocess.py --data_dir='raw' --output_dir='output_fz_extended' --extend=True --select_ch='EEG Fpz-Cz'```
		- [optional] Non extended : ```python preprocess.py --data_dir='raw' --output_dir='output_fz' --select_ch='EEG Fpz-Cz'```

5. Switch to earlier conda env
	- ```conda activate rest```
	
