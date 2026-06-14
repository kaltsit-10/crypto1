#!/usr/bin/env python3
"""Master monitor: periodically check all worker saves, write best to protected file."""
import numpy as np, os, time, glob

SAVE_DIR = '/home/linux/PycharmProjects/pythonProject/crypto1/p4_saves'
BEST_PATH = '/home/linux/PycharmProjects/pythonProject/crypto1/p4_gpu_solution.npy'
BEST_LOCKED = '/home/linux/PycharmProjects/pythonProject/crypto1/p4_solution_protected.npy'

print("Master monitor started")
while True:
    best_li = 999; best_file = ''
    for f in sorted(glob.glob(os.path.join(SAVE_DIR, '*.npy'))):
        try:
            s = np.load(f); v = s[:120]; u = s[120:]
            li = max(int(np.max(np.abs(v))), int(np.max(np.abs(u))))
            if li < best_li:
                best_li = li; best_file = f
        except:
            pass
    if best_li < 999:
        try:
            best_data = np.load(best_file)
            np.save(BEST_PATH, best_data)
            np.save(BEST_LOCKED, best_data)
            os.chmod(BEST_LOCKED, 0o444)
            if best_li <= 18:
                print(f"SAVED linf={best_li} from {best_file}")
        except Exception as e:
            print(f"Save error: {e}")
    time.sleep(15)
