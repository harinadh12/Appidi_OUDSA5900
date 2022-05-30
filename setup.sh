# to get started with tensorflow federated run the following commands in order

#download minconda3 
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh 

#run shell file to install minconda3 base env
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh

#restart the shell to activate base env

#create new tff env for project
conda create --name tff --clone base

# install tensorflow-federated using pip
pip install --upgrade tensorflow-federated


