import subprocess, sys, os, time

# Kill anything on port 8000 (all listeners, not just the first)
r = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
killed = set()
for line in r.stdout.splitlines():
    if ':8000' in line and 'LISTENING' in line:
        pid = line.strip().split()[-1]
        if pid not in killed:
            subprocess.run(['taskkill', '/PID', pid, '/F'])
            print('Killed PID', pid)
            killed.add(pid)
if killed:
    time.sleep(1)
else:
    print('Port 8000 is free')

# Start server. Flush stdout/stderr first — os.execv replaces the process
# image without flushing Python's buffers, so prints above would be lost.
sys.stdout.flush()
sys.stderr.flush()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.execv(sys.executable, [sys.executable, '-m', 'hallucination_guard.api.app'])
