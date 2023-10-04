1. Open terminal
2. Set up anaconda virtual environment
   1. If you need to install anaconda:
      1. `wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh`
      2. `bash Anaconda3-2023.09-0-Linux-x86_64.sh`
      3. Refresh terminal (close and reopen)
      4. `conda update conda`
      5. `rm Anaconda3-2023.09-0-Linux-x86_64.sh`
   2. `conda update conda`
   3. `conda create --name dimers python=3.9`
   4. `conda activate dimers`
   5. `pip install --upgrade pip`
   6. `conda install numpy scipy pandas matplotlib seaborn`
   7. `pip install --upgrade eqtk`
   8. `pip install pymoo==0.5.0`
3. Set up dimer code:
   1. Go to a directory where you want to install the code, e.g. `cd ~/code_projects`
   2. If you need to install git:
      1. If you need to install brew:
         1. `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
      2. `brew install git`
   3. `git clone https://github.com/BenEmert/dimer_computations_colab.git`
   4. `cd dimer_computations_colab/run_scripts`
   5. `git checkout paper_v0`
   6. `python michael_demo_v0.py --m 4 --nKs 2 --nTargets 2`
