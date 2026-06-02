import subprocess
import time
import argparse

def sync(dataset='*'):
    print('Syncing...')
    start = time.time()
    with subprocess.Popen(
        f'wandb sync ./exp/{dataset}/**/wandb/offline-run-*',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, shell=True
    ) as proc:
        for line in proc.stdout:
            print(line, end='')
    elapsed = time.time() - start
    print(f'Sync took {elapsed:.2f}s')
    return elapsed

def sync_daemon(dataset='*', interval_sec=30):
    while True:
        elapsed = sync(dataset)
        sleep_time = max(0, interval_sec - elapsed)
        time.sleep(sleep_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='*')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    sync_daemon(args.dataset, args.interval)