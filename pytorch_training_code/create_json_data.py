import numpy as np
from dataset.ei_dataset import EdgeImpulseDataset
import json
from tqdm import tqdm
import argparse
import os
import time, hmac, hashlib
import requests

if __name__ == '__main__':
    # add argrument for output directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='generated_json_data')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    validation_dataset = EdgeImpulseDataset('data/EI_dataset/testing', split='all')
    emptySignature = ''.join(['0'] * 64)

    template = data = {
        "protected": {
            "ver": "v1",
            "alg": "HS256",
            "iat": time.time() # epoch time, seconds since 1970
        },
        "signature": emptySignature,
        "payload": {
            "device_name": "ac:87:a3:0a:2d:1b",
            "device_type": "DISCO-L475VG-IOT01A",
            "interval_ms": 1,
            "sensors": [
                { "name": "axis0", "units": " " },
            ],
            "values": []
        }
    }
    
    for i in tqdm(range(len(validation_dataset))):
        data, label = validation_dataset[i]
        # save the data as json, field name is 'axis0', file name is {label}.{index}.json
        data_dict = template.copy()
        data_dict['payload']['values'] = data.tolist()
        file_name = f'{label}.{i}.json'
        file_name = os.path.join(args.output_dir, file_name)
        with open(file_name, 'w') as f:
            json.dump(data_dict, f)
            
        
        