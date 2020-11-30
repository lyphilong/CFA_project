#!/usr/bin/env python


import pandas as pd  
import subprocess

path = './dataset/url.csv'


list_file = pd.read_csv('./RAISE_257.csv')
new_url = pd.DataFrame(list_file['NEF'])
new_url.to_csv(path, header=False, index=False)

cmd = [ 'wget', '-i', path, '-P', './dataset']
output = subprocess.run(cmd)

cmd = [ 'rm', path]
output = subprocess.run(cmd)


