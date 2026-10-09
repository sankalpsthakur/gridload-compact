import shutil
def free_gb(path="/"):
    return shutil.disk_usage(path).free / 1e9
def disk_ok(min_gb=34.0):
    return free_gb() >= min_gb
