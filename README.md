# Brain Tumor Detection using yolov5

This is a streamlit app helps you to detect your brain tumor from MRI images.

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

Download and install [Python 3.10.0](https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe)

### Installation

1. First check python version in terminal, if it's 3.10.0 then go ahead.
   ```sh
   python --version 
   ```

2. Go to folder where you want to clone this repository and use below command to clone this repo to your local machine.
   ```sh
   git clone https://github.com/ipiyushvaghela/BrainTumorDetectionYOLOv5.git
   ```
3. Create virtual Environment 
   ```sh
   python -m venv venv4braintumor --system-site-packages
   ```

   To Activate our virtual environment we use 
   ```sh
   venv4braintumor/Script/Activate.bat
   ```
4. Install packages using requirements.txt
   ```sh
   pip install -r requirements.txt
   ```
4. Now check if streamlit is installed properly or not.
   ```sh
   streamlit version
   ```

## Usage

1. change paths of model in `app.py` file.
2. Run below command in terminal to run streamlit application. 
   ```sh
   streamlit run app.py
   ```
Now we are all set to go. just upload the image of any given class and it will classify that image and also it gives other 5 possible classes from which image could belong to.


![App Screenshot](https://github.com/ipiyushvaghela/static/blob/main/braintumor.png?raw=true)


## Info about other folders
1. Folder named `yolov5_setup` contains 2 files from which `braintumor_temp` can be used if you want to setup yolov5 to your local machine. other file is `dataset.yaml` you can edit this file and put straight into yolov5 cloned directory.
2. Folder named `models` contains all the trained models.



## Contact

Piyush Vaghela - [@ipiyushvaghela](https://twitter.com/ipiyushvaghela) - ipiyushvaghela@gmail.com

Project Link -  [github.com/ipiyushvaghela/BrainTumorDetectionYOLOv5](https://github.com/ipiyushvaghela/BrainTumorDetectionYOLOv5.git)