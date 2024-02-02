import os
import sys
import numpy as np
import re

def process_results(log_dir, summary_file, cdr_types, folds, name):
    log_dir = os.path.abspath(log_dir)

    # Extract alpha value from name
    # alpha = name.split('_alpha_')[-1].split('_')[0]

    # Extracting "iter0", "iter1", etc. part from SUMMARY_FILE
    match = re.search(r'iter\d+', summary_file)
    if match:
        iter_num = match.group()
        print(f"Extracted iter part: {iter_num}")
    else:
        iter_num = 0
        print("Iter part not found in the filename.")
    
    for cdr in cdr_types:
        acc_means = []
        tvd_means = []
        cosim_means = []

        for fold in folds:
            file_path = os.path.join(log_dir, f'cdrh{cdr}', f'fold_{fold}', name, summary_file)

            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        metrics = {pair.split('=')[0].strip(): float(pair.split('=')[1].strip()) for pair in line.split('|')}
                        acc_means.append(metrics.get("acc_mean", 0))
                        tvd_means.append(metrics.get("tvd_mean", 0))
                        cosim_means.append(metrics.get("cosim_mean", 0))
            else:
                print(f"File not found: {file_path}")

        # Compute average and standard deviation using numpy
        acc_mean_avg = np.mean(acc_means)
        acc_mean_stddev = np.std(acc_means)
        tvd_mean_avg = np.mean(tvd_means)
        tvd_mean_stddev = np.std(tvd_means)
        cosim_mean_avg = np.mean(cosim_means)
        cosim_mean_stddev = np.std(cosim_means)

        # Create the output file
        output_file = os.path.join(log_dir, f'cdrh{cdr}', f"average_results_{name}_{iter_num}.txt")
        
        with open(output_file, 'w') as out:
            out.write(f"acc_mean_avg={acc_mean_avg:.4f} | acc_mean_stddev={acc_mean_stddev:.4f}\n")
            out.write(f"tvd_mean_avg={tvd_mean_avg:.4f} | tvd_mean_stddev={tvd_mean_stddev:.4f}\n")
            out.write(f"cosim_mean_avg={cosim_mean_avg:.4f} | cosim_mean_stddev={cosim_mean_stddev:.4f}\n")

        print(f"Processed results for cdrh{cdr} written to: {output_file}")

if __name__ == "__main__":
    LOG_DIR = sys.argv[1]
    SUMMARY_FILE = sys.argv[2]
    cdr_types = list(map(int, sys.argv[3].split(',')))
    folds = list(map(int, sys.argv[4].split(',')))
    NAME = sys.argv[5]
    
    process_results(LOG_DIR, SUMMARY_FILE, cdr_types, folds, NAME)
