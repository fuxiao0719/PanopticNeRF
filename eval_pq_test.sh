python run.py --type process_json --cfg_file configs/panopticnerf_test.yaml use_stereo False
python run.py --type process_json_gt --cfg_file configs/panopticnerf_test.yaml use_stereo False 
python run.py --type eval_pq --cfg_file configs/panopticnerf_test.yaml use_stereo False
