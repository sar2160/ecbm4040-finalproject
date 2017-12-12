
from __future__ import print_function

import os
import zipfile
import subprocess


def getCityImages(city, dest_directory = 'imagery/'):
    filename =  city + '.zip'
    if not os.path.isfile(dest_directory+filename):
        cmd = 'gsutil cp gs://sar-dl-store/' + '"' + filename + '" ' + dest_directory
        print('Executing: ') ; print(cmd)
        returned = subprocess.call(cmd, shell=True)
        print(returned)

    try:
        zip_ref = zipfile.ZipFile(dest_directory+filename, 'r')
        zip_ref.extractall(dest_directory)
        zip_ref.close()
        print('Extracted image files for: ' + city)

    except:
        print('File not found or unzip error')
