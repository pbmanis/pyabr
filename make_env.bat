Rem @echo off
set ENVNAME=pyabr_venv
set PSPATH=%cd%
py -3.10 -m venv %ENVNAME%
set ACT=C:\%PSPATH%\%ENVNAME%\Scripts\activate.bat
set DEACT=C:\%PSPATH%\%ENVNAME%\Scripts\deactivate.bat
%ACT%
python --version

python -m pip install --upgrade pip 
Rem  be sure pip is up to date in the new env.
pip3 install wheel  
Rem seems to be missing (note singular)
pip3 install cython
Rem # if requirements.txt is not present, create:
Rem # pip install pipreqs
Rem # pipreqs

Rem See note about how to get pyaudio in the requirements.txt file
pip3 install -r requirements.txt
# pip3 install downloads\\PyAudio-0.2.14-cp310-cp310-win_amd64.whl
pip3 install pyaudio==0.2.14

python setup.py develop

Rem Should always run test afterwards.
python tests/play_test_sounds.py

pause