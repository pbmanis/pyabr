ENVNAME="pyabr_venv"
python3.10 -m venv $ENVNAME
source $ENVNAME/bin/activate

pip3 install --upgrade pip  # be sure pip is up to date in the new env.
pip3 install wheel  # seems to be missing (note singular)
pip3 install cython
#
# #Then:
#
pip3 install -r requirements.txt
source $ENVNAME/bin/activate


source $ENVNAME/bin/activate
python --version
python setup.py develop

