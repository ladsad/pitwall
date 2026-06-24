import sys
import os
import re
import subprocess
from dotenv import load_dotenv

load_dotenv('.env.local')

def run_prediction(event, round_num):
    with open('config.py', 'r') as f:
        config_data = f.read()
    
    config_data = re.sub(r'EVENT\s*=\s*".*"', f'EVENT = "{event}"', config_data)
    config_data = re.sub(r'ROUND_NUMBER\s*=\s*\d+', f'ROUND_NUMBER = {round_num}', config_data)
    
    with open('config.py', 'w') as f:
        f.write(config_data)
        
    subprocess.run([sys.executable, 'notebooks/06_predict.py'], check=True)

if __name__ == '__main__':
    run_prediction('Canadian Grand Prix', 5)
    run_prediction('Barcelona Grand Prix', 7)
    print("Successfully predicted and uploaded Round 5 and 7")
