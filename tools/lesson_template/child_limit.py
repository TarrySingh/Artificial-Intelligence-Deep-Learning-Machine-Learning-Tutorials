# The block a lesson that RUNS A COMPILED PROGRAM carries, in addition to the COMMONS TIME LIMIT block of
# lesson.py in this folder (C and C++ lessons only). Copy it unchanged, once, above the first function that
# runs the student's binary, and run that binary, and `make test`, through `_run_child` (or, for a program that
# is fed a stream, under `with _Ceiling(proc):`) instead of `subprocess.run`. internal/retrofit_guard.py
# inserts it; tools/verify_all.py fails a lesson whose copy differs.
#
# Why. The time limit bounds how long a runaway answer RUNS, not how much it takes. Compiled code fills
# memory several times faster than Python: a C++ `push_back` loop grows by about 3 GB a second, so the
# 10-15 s that a demo cell is allowed would be 30-50 GB, which is swap on a Mac and the kernel's
# out-of-memory killer on an 8 GiB Colab or Binder machine (it may take the notebook's kernel with it).
# The program runs in a process group of its own and the whole group, `make` and the binary it started
# included, is stopped as soon as the group's resident memory passes `_CHILD_MIB`. The check fails with a
# message that says so, the notebook goes on, nothing is left running. The group is looked at every
# 0.05 s, so the peak is the ceiling plus what the program can allocate in 0.05 s (about 0.17 GB for a
# push_back loop). What a program PRINTS is bounded too: the notebook keeps every line it reads, and a loop
# that prints for ever fills the notebook's memory at the speed of a pipe (1-3 GB a second), which is
# where the child's own ceiling cannot see it; past `_CHILD_OUTPUT_MIB` the program is stopped.
# --- COMMONS CHILD LIMIT v2 --- a program a check starts has a memory ceiling too
import os, signal, subprocess, sys, threading

_CHILD_MIB = 1024      # how much memory a program started by a check, and what it starts, may use before it is stopped
_CHILD_OUTPUT_MIB = 64  # how much it may print (stdout and stderr together) before it is stopped


def _group_mib(pgid: int) -> float:
    """Resident memory, in MiB, of every process in the process group `pgid` (Linux: /proc; macOS: ps)."""
    pages = kib = 0
    try:
        if os.path.isdir("/proc/self"):
            for entry in os.listdir("/proc"):
                if entry.isdigit():
                    try:
                        with open(f"/proc/{entry}/stat") as f:
                            fields = f.read().rsplit(")", 1)[1].split()
                        if int(fields[2]) == pgid:
                            with open(f"/proc/{entry}/statm") as f:
                                pages += int(f.read().split()[1])
                    except (OSError, ValueError, IndexError):
                        continue
            return pages * os.sysconf("SC_PAGE_SIZE") / 2**20
        listing = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True, timeout=10).stdout
        for row in listing.splitlines():
            parts = row.split()
            if len(parts) == 2 and parts[0] == str(pgid):
                kib += int(parts[1])
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0
    return kib / 1024


class _Ceiling:
    """`with _Ceiling(proc):` around the wait for a program started with `start_new_session=True`: stops its whole
    process group if the group's memory passes `_CHILD_MIB`, and on the way out (an exception, a time limit
    arriving, or a normal end) leaves nothing of the group running. Afterwards `.check(name)` raises the
    AssertionError that says why, if the ceiling is what stopped it."""

    def __init__(self, proc):
        self.proc, self.over, self._done = proc, [], threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def kill(self):
        try:
            if hasattr(os, "killpg"):
                os.killpg(self.proc.pid, signal.SIGKILL)
            else:                                            # Windows: no process groups, so the program alone
                self.proc.kill()
        except OSError:
            pass

    def _watch(self):
        while not self._done.wait(0.05):
            used = _group_mib(self.proc.pid)
            if used > _CHILD_MIB:
                self.over.append(used)
                self.kill()
                return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._done.set()
        self._thread.join(2)
        self.kill()
        if exc_type is not None:
            self.proc.wait()
        return False

    def check(self, name: str) -> None:
        if self.over:
            raise AssertionError(f"{os.path.basename(str(name))} was stopped after it used over {_CHILD_MIB} MiB of memory "
                                 f"(it had reached {self.over[0]:.0f} MiB): a loop in your C code that keeps allocating or "
                                 "appending never stops; give every loop a cap")
        if getattr(self, "printed", 0) > _CHILD_OUTPUT_MIB * 2**20:
            raise AssertionError(f"{os.path.basename(str(name))} was stopped after it printed over {_CHILD_OUTPUT_MIB} MiB: a loop "
                                 "in your C code that prints on every turn never stops; give every loop a cap")


def _run_child(cmd, *, cwd=None, env=None, input=None, stdin=None, timeout=None):
    """`subprocess.run(cmd, capture_output=True, text=True, ...)` for a program that runs the student's code, under
    the memory ceiling above and a bound on what it prints: an AssertionError that says so if it passes either,
    `subprocess.TimeoutExpired` if `timeout` passes (as subprocess.run raises it), and nothing left running in its
    group whatever happens. Returns a CompletedProcess."""
    if not (hasattr(os, "killpg") and hasattr(signal, "SIGKILL")):          # Windows: no process groups, no ceiling
        return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=timeout,
                              **({"input": input} if input is not None else {"stdin": stdin}))
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE if input is not None else stdin,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    chunks, printed = ([], []), [0]

    def read(pipe, into):
        for block in iter(lambda: pipe.read1(65536), b""):
            into.append(block)
            printed[0] += len(block)
            if printed[0] > _CHILD_OUTPUT_MIB * 2**20:
                ceiling.printed = printed[0]
                ceiling.kill()
                break

    with _Ceiling(proc) as ceiling:
        readers = [threading.Thread(target=read, args=(proc.stdout, chunks[0]), daemon=True),
                   threading.Thread(target=read, args=(proc.stderr, chunks[1]), daemon=True)]
        for r in readers:
            r.start()
        if input is not None:
            try:
                proc.stdin.write(input.encode())
                proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired(cmd, timeout) from None
        for r in readers:
            r.join(5)
    ceiling.check(cmd[0])
    import locale
    out, err = (b"".join(c).decode(locale.getpreferredencoding(False), "replace").replace("\r\n", "\n").replace("\r", "\n")
                for c in chunks)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
# --- END COMMONS CHILD LIMIT ---
