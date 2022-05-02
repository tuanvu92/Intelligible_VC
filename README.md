# Intelligible VC: Increase speech intelligibility by mimicking voice of professional announcer using voice conversion
Author: Ho Tuan Vu - Japan Advanced Institute of Science and Technology 

## Abstract
In most of practical scenarios, the announcement system must deliver speech messages in a noisy environment, in which the background noise cannot be cancelled out. The local noise reduces speech intelligibility and increases listening effort of the listener, hence hamper the effectiveness of announcement system. There has been reported that voices of professional announcers are clearer and more comprehensive than that of non-expert speakers in noisy environment. This finding suggests that the speech intelligibility might be related to the speaking style of professional announcer, which can be adapted using voice conversion method. Motivated by this idea, this paper proposes a speech intelligibility enhancement in noisy environment by applying voice conversion method on non-professional voice. We discovered that the professional announcers and non-professional speakers are clusterized into different clusters on the speaker embedding plane. This implies that the speech intelligibility can be controlled as an independent feature of speaker individuality. To examine the advantage of converted voice in noisy environment, we experimented using test words masked in pink noise at different SNR levels. The results of objective and subjective evaluations confirm that the speech intelligibility of converted voice is higher than that of original voice in low SNR conditions.


##1. Installation
Install using pip:
> pip install https://github.com/tuanvu92/Intelligible_VC.git

Clone this repo: 
> git clone https://github.com/tuanvu92/Intelligible_VC.git
> cd IntellgibleVC
> pip install -r requirement.txt

## 2. Data
Please get access to AIS-Lab internal server //refresh/share/database and download ATR data. 

To run preprocessing, modify the __CORPUS_PATH__ and __OUTPATH__ in preprocess_data.py file.

Otherwise, already processed data can be downloaded here: 
https://jstorage-2018.jaist.ac.jp/s/ecL69LijMZLJjDe

Please send email to tuanvu@jaist.ac.jp for password.

Extract the zip file and place all files under **data/** folder.

## 3. Run training
> python distributed.py -f train.py -c config/train_config.yml

## 4. Demo notebook
Install Jupyter notebook: https://jupyter.org/install

Start Jupyter notebook server:
> cd IntelligibleVC
> 
> jupyter notebook

Start browser and open the address: http://localhost:9999

Locate the demo notebook in **notebooks/demo.ipynb**







