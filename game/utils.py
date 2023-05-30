from tqdm import tqdm

def mean(data):
    return sum(data) / len(data)

def _print_redirect(f, msg):
    tqdm.write(msg)
    if f: f.write(msg+"\n")