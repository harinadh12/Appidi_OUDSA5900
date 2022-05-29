#install python 3.9
sudo apt-get update -y
sudo apt-get install python3.9

#change alternatives
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2

#check python version
python --version
#3.9.6

# to update pip
sudo apt install python3-pip
sudo apt install python3.9-distutils
python3 -m pip install --upgrade pip


pip install --quiet --upgrade tensorflow-federated
pip install --quiet --upgrade nest-asyncio
