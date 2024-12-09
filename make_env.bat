Rem @echo off
set ENVNAME=pyabr_venv
set PSPATH=%cd%
py -3.12 -m venv %ENVNAME%
set ACT=C:\%PSPATH%\%ENVNAME%\Scripts\activate.bat
set DEACT=C:\%PSPATH%\%ENVNAME%\Scripts\deactivate.bat
%ACT%
python --version

py -m pip install --upgrade pip 
Rem  be sure pip is up to date in the new env.
py -m pip install wheel  
Rem seems to be missing (note singular)
py -m pip install cython
Rem # if requirements.txt is not present, create:
Rem # pip install pipreqs
Rem # pipreqs

Rem See note about how to get pyaudio in the requirements.txt file
py -m pip install -r requirements.txt
Rem py -m pip install downloads\\PyAudio-0.2.14-cp310-cp310-win_amd64.whl
py -m pip install pyaudio==0.2.14

python setup.py develop

Rem Should always run test afterwards.
python tests/play_test_sounds.py pip

pause