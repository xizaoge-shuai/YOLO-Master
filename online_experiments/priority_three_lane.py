"""Adopt only the SKU supervisor; overlap its fixed queue with SKU Detail25.

No trainer, source anchor, hyperparameter, dataset, seed, or output changes.
The two old children and the VOC supervisor remain untouched. Linux only.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_experiments import priority_single_seed_campaign as original
from online_experiments.crossdataset_campaign import GPU_LOCK_ROOT, gpu_memory, job_lock

INT8, ADAPTER, DETAIL = "cache-reuse-c1-int8", "vit-adapter-cache25", "detail25"


def process(pid):
    try:
        folder = Path("/proc") / str(pid)
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "uid": folder.stat().st_uid, "state": fields[0], "ppid": int(fields[1]),
                "start_ticks": int(fields[19]), "argv": (folder / "cmdline").read_bytes().decode(errors="surrogateescape").rstrip("\0").split("\0")}
    except OSError:
        return None


def same_process(expected, current):
    if current is None or any(current.get(k) != expected.get(k) for k in ("pid", "uid", "start_ticks", "argv")):
        raise RuntimeError("Process identity changed; no signal or restart is authorized")


def same_lifetime(expected, current):
    if current is None or any(current.get(k) != expected.get(k) for k in ("pid", "uid", "start_ticks")):
        raise RuntimeError("Process lifetime changed while waiting for exit")


def libc_pidfd(name, *values):
    """Use glibc's checked pidfd binding when this Python omits it."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, name, None)
    if function is None:
        raise RuntimeError("No protected pidfd binding available: " + name)
    function.restype = ctypes.c_int
    if name == "pidfd_open":
        function.argtypes = (ctypes.c_int, ctypes.c_uint)
        result = function(values[0], 0)
    else:
        function.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint)
        result = function(values[0], values[1], None, 0)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def open_pidfd(pid):
    builtin = getattr(os, "pidfd_open", None)
    return builtin(pid) if callable(builtin) else libc_pidfd("pidfd_open", pid)


def send_pidfd(fd, sig):
    builtin = getattr(signal, "pidfd_send_signal", None)
    return builtin(fd, sig) if callable(builtin) else libc_pidfd("pidfd_send_signal", fd, sig)


def pidfd_capability():
    # Signal 0 only checks permission/liveness; it delivers no signal.
    fd = open_pidfd(os.getpid())
    try:
        send_pidfd(fd, 0)
    finally:
        os.close(fd)


def signal_one(expected):
    # pidfd prevents a PID-reuse race between the identity check and SIGTERM.
    fd = open_pidfd(expected["pid"])
    try:
        same_process(expected, process(expected["pid"]))
        send_pidfd(fd, signal.SIGTERM)
    finally:
        os.close(fd)


def supervisor_argv(argv, metadata_argv):
    script = str(ROOT / "online_experiments/priority_single_seed_campaign.py")
    return bool(metadata_argv and metadata_argv[0] == script and
                argv in ([sys.executable, "-u", *metadata_argv], [sys.executable, *metadata_argv]))


def check(args):
    if args.dataset != "sku110k" or sys.platform != "linux":
        raise RuntimeError("Requires SKU-only Linux pidfd support")
    pidfd_capability()
    saved = original.read(args.job / "pipeline-plan.json")
    if original.plan(args) != saved:
        raise RuntimeError("Frozen original plan/source/cache changed")
    metadata, active = original.read(args.job / "worker.json"), original.read(args.job / "active-child.json")
    old, child = process(args.old_supervisor), process(args.adopt_child)
    if old is None or child is None or metadata["pid"] != args.old_supervisor or active["pid"] != args.adopt_child:
        raise RuntimeError("Expected exact old supervisor and INT8 child are not present")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if metadata["boot_id"] != boot or old["uid"] != os.getuid() or child["uid"] != os.getuid():
        raise RuntimeError("Boot or process owner differs")
    if not supervisor_argv(old["argv"], metadata["argv"]):
        raise RuntimeError("Supervisor command differs from recorded SKU worker")
    if old["argv"][old["argv"].index("--job") + 1] != str(args.job):
        raise RuntimeError("Supervisor is not the exact canonical SKU job")
    if child["ppid"] != old["pid"] or active["action"] != "train" or active["method"] != INT8:
        raise RuntimeError("Only the exact live INT8 child may be adopted")
    if child["argv"] != original.command_for(args, "train", INT8) or active["argv"] != child["argv"]:
        raise RuntimeError("Adopted training command differs from frozen command")
    if old["state"] == "Z" or child["state"] == "Z":
        raise RuntimeError("Supervisor or adopted worker has already exited")
    check_pending(args, child)
    return {"supervisor": old, "child": child, "boot_id": boot, "plan_sha256": original.sha(args.job / "pipeline-plan.json"),
            "rollback_argv": [sys.executable, *metadata["argv"]], "policy": "Never kill child/group; preserve checkpoints; no automatic retry"}


def check_pending(args, child):
    completed = original.read(args.job / "completed-methods.json")
    if [item["method"] for item in completed] != list(original.ordered_methods(args)[:4]):
        raise RuntimeError("Expected the four original baseline endpoints before INT8")
    if any(not (original.output_for(args, item["method"]) / ("DONE.json" if item["method"] in original.DIRECT else "DONE")).is_file() for item in completed):
        raise RuntimeError("Prior baseline endpoint completion marker missing")
    for method in (ADAPTER, DETAIL):
        if original.output_for(args, method).exists():
            raise RuntimeError("Refuse existing pending-method output: " + method)
    expected = str(original.output_for(args, INT8))
    found = [p for p in training_processes() if "--output" in p["argv"] and p["argv"][p["argv"].index("--output") + 1] == expected]
    if [p["pid"] for p in found] != [child["pid"]]:
        raise RuntimeError("Live adopted output does not have exactly one owner")


def verify_recovery_processes(evidence):
    old, child = evidence["supervisor"], evidence["child"]
    current = process(old["pid"])
    if current is not None and current.get("state") != "Z":
        raise RuntimeError("Recovery requires the old supervisor to have exited")
    current = process(child["pid"])
    same_process(child, current)
    if current.get("state") == "Z":
        raise RuntimeError("Recovery requires the exact adopted child still alive")


def check_recovery(args):
    if args.dataset != "sku110k" or sys.platform != "linux":
        raise RuntimeError("Requires SKU-only Linux recovery")
    saved = original.read(args.job / "pipeline-plan.json")
    if original.plan(args) != saved:
        raise RuntimeError("Frozen original plan/source/cache changed")
    evidence = original.read(args.sidecar / "handoff.json")
    previous_plan = original.read(args.sidecar / "plan.json")
    digest = original.sha(args.job / "pipeline-plan.json")
    if evidence["plan_sha256"] != digest or previous_plan["frozen_pipeline_plan_sha256"] != digest:
        raise RuntimeError("Recovery handoff and original plan are not the same sealed job")
    if evidence["supervisor"]["pid"] != args.old_supervisor or evidence["child"]["pid"] != args.adopt_child:
        raise RuntimeError("Recovery PID arguments differ from persisted handoff")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if evidence["boot_id"] != boot or evidence["child"]["uid"] != os.getuid():
        raise RuntimeError("Recovery boot or adopted owner differs")
    verify_recovery_processes(evidence)
    metadata, active = original.read(args.job / "worker.json"), original.read(args.job / "active-child.json")
    if metadata["pid"] == args.old_supervisor:
        if not supervisor_argv(evidence["supervisor"]["argv"], metadata["argv"]):
            raise RuntimeError("Canonical old worker metadata differs from handoff")
    elif metadata.get("scheduler") != "priority-three-lane-v1" or metadata.get("handoff_file") != str(args.sidecar / "handoff.json"):
        raise RuntimeError("Canonical worker is not the old supervisor or this explicit adoption")
    elif (current := process(metadata["pid"])) is not None and current.get("state") != "Z":
        raise RuntimeError("An adopting supervisor is still alive")
    child = evidence["child"]
    if active["pid"] != child["pid"] or active["action"] != "train" or active["method"] != INT8:
        raise RuntimeError("Canonical active child differs from persisted INT8 handoff")
    if active["argv"] != child["argv"] or child["argv"] != original.command_for(args, "train", INT8):
        raise RuntimeError("Recovery child command differs from frozen trainer")
    check_pending(args, child)
    return evidence


def scheduler_plan(args, evidence):
    return {"schema": "priority-three-lane-v1", "dataset": "sku110k", "job": str(args.job),
            "frozen_pipeline_plan_sha256": evidence["plan_sha256"], "adopted_child": evidence["child"],
            "user_override": "Allow three existing core methods concurrently, single seed, unchanged full-data protocol",
            "methods": {"adopt": INT8, "baseline": ADAPTER, "extra": DETAIL},
            "epochs": 100, "batch": 8, "imgsz": 640, "seed": 0, "gpu": 0,
            "extra_admission": {"free_gpu_mib": 14336, "available_host_mib": 12288, "max_our_training_processes": 3},
            "source_sha256": {name: original.sha(ROOT / name) for name in (
                "online_experiments/priority_three_lane.py", "online_experiments/test_priority_three_lane.py")},
            "source_policy": "Unmodified frozen original sources verified by original.plan; only scheduling changes",
            "failure_policy": "Preserve all checkpoints and live children; no automatic retry; other lane continues"}


def handoff(args, evidence):
    old, child = evidence["supervisor"], evidence["child"]
    same_process(old, process(old["pid"]))
    same_process(child, process(child["pid"]))
    original.write(args.sidecar / "handoff.json", dict(evidence, state="INTENT", time_unix=time.time()))
    signal_one(old)
    deadline = time.monotonic() + 20
    while (current := process(old["pid"])) is not None and current.get("state") != "Z":
        # cmdline legitimately clears during exit, after the one verified signal.
        same_lifetime(old, current)
        if time.monotonic() > deadline:
            raise RuntimeError("Old supervisor did not exit; child is preserved, do not start extra work")
        time.sleep(0.2)
    current = process(child["pid"])
    same_process(child, current)
    if current.get("state") == "Z":
        raise RuntimeError("Adopted child exited during handoff; inspect endpoint before recovery")
    original.write(args.sidecar / "handoff.json", dict(evidence, state="SUPERVISOR_REPLACED_CHILD_ALIVE", time_unix=time.time()))


def wait_adopted(args, child):
    while (current := process(child["pid"])) is not None and current.get("state") != "Z":
        same_lifetime(child, current)
        if current.get("argv") != [""]:
            same_process(child, current)
        time.sleep(10)
    if not (original.output_for(args, INT8) / "DONE").is_file():
        raise RuntimeError("Adopted INT8 exited without sealed DONE; no automatic restart")


def training_processes():
    names = {"crossdataset_suite.py", "sku_official_loader_suite.py", "direct_baseline_suite.py"}
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        record = process(int(path.name))
        if record and record["uid"] == os.getuid() and record["state"] != "Z" and "train" in record["argv"]:
            if any(Path(value).name in names for value in record["argv"]):
                found.append(record)
    return found


def host_memory():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) // 1024


def can_admit(slot, occupied, workers, free_mib, host_mib):
    required = 14336 if slot == 2 else 8192 + (8192 if occupied else 0)
    return occupied < 3 and workers < 3 and free_mib >= required and host_mib >= 12288


@contextlib.contextmanager
def lease(args, lane, state):
    import fcntl
    GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    selected = None
    while selected is None:
        candidates = []
        with (GPU_LOCK_ROOT / "gpu0-stages.lock").open("a") as gate:
            try:
                fcntl.flock(gate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                busy, available = 0, {}
                for slot in range(3):
                    stream = (GPU_LOCK_ROOT / f"gpu0-slot{slot}.lock").open("a")
                    candidates.append(stream)
                    try:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        busy += 1
                    else:
                        available[slot] = stream
                slots = (2,) if lane == "extra" else (0, 1)
                free, host, workers = gpu_memory(0), host_memory(), len(training_processes())
                for slot in slots:
                    if slot in available and can_admit(slot, busy, workers, free, host):
                        selected = available[slot]
                        break
        for stream in candidates:
            if stream is not selected:
                stream.close()
        if selected is None:
            state.update(lane, "WAITING_RESOURCES")
            time.sleep(15)
    try:
        args.gpu_lock_fd = selected.fileno()
        yield
    finally:
        del args.gpu_lock_fd
        selected.close()


class State:
    def __init__(self, args):
        self.args, self.lock, self.lanes = args, threading.RLock(), {}
        self.detail_done = threading.Event()
        self.detail_error = None
        self.errors = []
        self.completed = original.read(args.job / "completed-methods.json")

    def complete(self, method, archive):
        with self.lock:
            self.completed = [item for item in self.completed if item["method"] != method]
            self.completed.append({"method": method, "output": str(original.output_for(self.args, method)), "archive": archive})
            order = list(original.ordered_methods(self.args))
            self.completed.sort(key=lambda item: order.index(item["method"]))
            original.write(self.args.job / "completed-methods.json", self.completed)

    def update(self, lane, state, **extra):
        with self.lock:
            self.lanes[lane] = {"state": state, "time_unix": time.time(), **extra}
            record = {"schema": "priority-three-lane-v1", "pid": os.getpid(), "lanes": self.lanes,
                      "errors": self.errors, "timing": "Shared GPU; observational, not controlled speed evidence"}
            original.write(self.args.sidecar / "status.json", record)
            active = [value["method"] for value in self.lanes.values() if value.get("method") and value["state"] not in ("COMPLETED", "FAILED")]
            original.status(self.args, "RUNNING_THREE_LANE" if not self.errors else "PARTIAL_FAILURE", "SKU INT8 adoption + Adapter baseline + parallel Detail25", lanes=self.lanes,
                            completed=len(self.completed), planned=7, sidecar=str(self.args.sidecar),
                            method=self.lanes.get("baseline", {}).get("method"), parallel_methods=active)


def execute(args, action, method, state, lane):
    command = original.command_for(args, action, method)
    output = str(original.output_for(args, method))
    if action == "train" and any("--output" in p["argv"] and p["argv"][p["argv"].index("--output") + 1] == output for p in training_processes()):
        raise RuntimeError("Refuse duplicate training output: " + output)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HOME=str(ROOT.parent / "cache/huggingface"), GH_CONFIG_DIR=str(ROOT.parent / ".config/gh"),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1",
               TMPDIR=str(ROOT.parent / "tmp/paper-baselines"), PYTHONPATH=str(args.dependencies) + os.pathsep + str(ROOT))
    original.write(args.sidecar / "commands" / f"{action}-{method}.json", {"argv": command, "cwd": str(ROOT)})
    with (args.sidecar / f"{action}-{method}.log").open("a") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 pass_fds=(args.gpu_lock_fd,) if hasattr(args, "gpu_lock_fd") else ())
        state.update(lane, action.upper(), method=method, child_pid=child.pid, output=output)
        result = child.wait()
    if result:
        raise RuntimeError(f"{action} {method} exited {result}; preserve checkpoint, no retry")


def finish(args, method, state):
    execute(args, "validate", method, state, "extra" if method == DETAIL else "baseline")
    if not (original.output_for(args, method) / "DONE").is_file():
        raise RuntimeError("Validator lacks sealed DONE")
    archive = original.archive_case(args, method)
    state.complete(method, archive)
    return archive


def train_one(args, method, lane, state):
    with lease(args, lane, state):
        execute(args, "train", method, state, lane)
    finish(args, method, state)
    state.update(lane, "COMPLETED", method=method)


def wait_detail(state):
    state.detail_done.wait()
    if state.detail_error:
        raise RuntimeError("Detail lane failed; output preserved and not retried") from state.detail_error


def baseline(args, child, state):
    state.update("baseline", "ADOPTED_TRAINING", method=INT8, child_pid=child["pid"])
    wait_adopted(args, child)
    finish(args, INT8, state)
    train_one(args, ADAPTER, "baseline", state)
    wait_detail(state)


def run_thread(args, lane, evidence, state):
    try:
        if lane == "baseline":
            baseline(args, evidence["child"], state)
        else:
            train_one(args, DETAIL, "extra", state)
    except Exception as error:
        with state.lock:
            state.errors.append({"lane": lane, "traceback": traceback.format_exc()})
            if lane == "extra":
                state.detail_error = error
        state.update(lane, "FAILED", error=str(error), policy="Other lane continues; no retry")
    finally:
        if lane == "extra":
            state.detail_done.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "status"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--dataset-job", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--old-supervisor", type=int, default=641892)
    parser.add_argument("--adopt-child", type=int, default=1881349)
    parser.add_argument("--dependencies", type=Path, default=ROOT / ".research-deps/evidence-20261110/site")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--recover", action="store_true", help="Adopt the persisted child after a failed supervisor handoff; never signal")
    args = parser.parse_args()
    args.job, args.dataset_job, args.sidecar = (getattr(args, key).resolve() for key in ("job", "dataset_job", "sidecar"))
    args.dataset, args.first_detector = "sku110k", "yolo12-n"
    if args.action == "status":
        print(original.read(args.sidecar / "status.json"))
        return
    checker = check_recovery if args.recover else check
    evidence = checker(args)
    if args.action == "check":
        import json
        print(json.dumps(evidence, indent=2))
        return
    args.sidecar.mkdir(parents=True, exist_ok=True)
    with job_lock(args.sidecar):
        evidence = checker(args)
        original.write(args.sidecar / ("recovery-plan.json" if args.recover else "plan.json"), scheduler_plan(args, evidence))
        try:
            if not args.recover:
                handoff(args, evidence)
            with job_lock(args.job):
                if args.recover:
                    evidence = check_recovery(args)
                    original.write(args.sidecar / "recovery-handoff.json", dict(evidence, state="ADOPTED_WITHOUT_SIGNAL", time_unix=time.time()))
                state = State(args)
                worker = {"pid": os.getpid(), "argv": sys.argv, "started_unix": time.time(), "boot_id": evidence["boot_id"],
                          "handoff_file": str(args.sidecar / "handoff.json"), "scheduler": "priority-three-lane-v1"}
                original.write(args.sidecar / "worker.json", dict(worker, handoff=evidence))
                original.write(args.job / "worker.json", worker)
                threads = [threading.Thread(target=run_thread, args=(copy.copy(args), lane, evidence, state), name=lane) for lane in ("baseline", "extra")]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                if state.errors:
                    raise RuntimeError("One or more lanes failed; inspect sidecar status and preserved outputs")
                original.status(args, "SUCCEEDED", "All seven SKU seed0/final100 endpoints completed; Detail25 was parallelized")
        except Exception:
            failure = args.sidecar / (f"recovery-required-{time.time_ns()}.json" if args.recover else "recovery-required.json")
            original.write(failure, {"traceback": traceback.format_exc(), "adopted_child": evidence["child"],
                           "policy": "Inspect live workers and DONE first. Never start original supervisor while a sidecar child is alive.", "rollback_argv": evidence["rollback_argv"]})
            raise


if __name__ == "__main__":
    main()
