import subprocess, sys, os, time

# Kill anything on port 8000
r = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if ':8000' in line and 'LISTENING' in line:
        pid = line.strip().split()[-1]
        subprocess.run(['taskkill', '/PID', pid, '/F'])
        print('Killed PID', pid)
        time.sleep(1)
        break
else:
    print('Port 8000 is free')

# Start server
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.execv(sys.executable, [sys.executable, '-m', 'hallucination_guard.api.app'])
