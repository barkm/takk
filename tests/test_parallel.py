import os
import signal
import subprocess
import sys
import time

from isolated_sign_validation.parallel import parallel_map


def test_parallel_map_keeps_order_across_batches():
    items = list(range(-50, 50))
    assert list(parallel_map(abs, items, batch_size=7, chunksize=3)) == [abs(i) for i in items]


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def test_ctrl_c_stops_main_process_and_workers_promptly():
    script = (
        "import time\n"
        "from isolated_sign_validation.parallel import parallel_map\n"
        "for _ in parallel_map(time.sleep, [0.5] * 1000, max_workers=2):\n"
        "    print('result', flush=True)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
    assert process.stdout is not None
    pgid = process.pid
    try:
        process.stdout.readline()  # wait until the workers are running
        for _ in range(2):  # Ctrl-C twice in a terminal: the whole process group gets SIGINT each time
            os.killpg(pgid, signal.SIGINT)
            time.sleep(0.2)
        process.wait(timeout=5)
        deadline = time.time() + 5
        while process_group_alive(pgid) and time.time() < deadline:
            time.sleep(0.1)
        assert not process_group_alive(pgid), "worker processes kept running after Ctrl-C"
        assert process.returncode != 0
    finally:
        if process_group_alive(pgid):
            os.killpg(pgid, signal.SIGKILL)
