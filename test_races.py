#!/usr/bin/env python3
"""Race-condition tests for the run/terminate/event coordination.

Run with:  python3 -m unittest test_races -v

Covers:
  * terminate arriving while the run is still registered but Popen has not
    returned (controlled thread interleaving via a gate);
  * immediate re-run after termination: old child dead+reaped, old run_id
    can no longer control it, two live children never coexist;
  * late events after termination can never flip a terminal run back to an
    active state (white-box event application + end-to-end event flood);
  * normal pause/step/continue flow and duplicate/stale 409 rules;
  * terminating a runaway infinite loop after continue;
  * multiple starters parked on an un-reaped terminal run: admission after
    wake-up is re-decided against the then-current run, so a replacement
    run that is itself terminal-but-unreaped is never leapfrogged, an
    active replacement makes the other waiters fail with 409, and a reaping
    timeout is a 409 that leaves the current run untouched.
"""

import contextlib
import json
import os
import threading
import time
import unittest
from unittest import mock

import app as appmod


SIMPLE_CODE = "x = 1\ny = 2\nprint(x + y)\n"
LOOP_CODE = "while True:\n    pass\n"
FLOOD_CODE = "for i in range(1000000):\n    print(i)\n"


def wait_for(predicate, timeout=10.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class RaceTests(unittest.TestCase):
    def setUp(self):
        self.client = appmod.app.test_client()
        self.created = []  # every Popen handle created during this test
        self._real_popen = appmod.subprocess.Popen

        def tracking_popen(*args, **kwargs):
            proc = self._real_popen(*args, **kwargs)
            self.created.append(proc)
            return proc

        self._patch = mock.patch.object(appmod.subprocess, "Popen", tracking_popen)
        self._patch.start()
        # Start every test from a clean registry.
        with appmod.COND:
            appmod.CURRENT = None

    def tearDown(self):
        self._patch.stop()
        # Best-effort cleanup of any child left behind by a failing test.
        with appmod.COND:
            current = appmod.CURRENT
        if (current is not None and current.status in appmod.ACTIVE_STATES
                and current.proc_started):
            self.client.post("/api/terminate", json={"run_id": current.run_id})
        for proc in self.created:
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass

    # ---- helpers -------------------------------------------------------

    def state(self):
        return self.client.get("/api/state").get_json()

    def start_run(self, code=SIMPLE_CODE):
        res = self.client.post("/api/run", json={"code": code})
        self.assertEqual(res.status_code, 200, res.get_data(as_text=True))
        return res.get_json()["run_id"]

    def wait_paused(self, run_id):
        ok = wait_for(lambda: self.state()["status"] == "paused")
        self.assertTrue(ok, f"run {run_id} never paused: {self.state()}")

    def assert_proc_dead(self, proc):
        self.assertIsNotNone(proc.poll(), "child process was not reaped")
        with self.assertRaises(ProcessLookupError):
            os.kill(proc.pid, 0)

    # ---- 1. terminate during startup (controlled interleaving) ---------

    def test_terminate_during_startup_then_immediate_rerun(self):
        entered = threading.Event()
        release = threading.Event()
        _real_popen = self._real_popen

        def gated_popen(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(timeout=10), "gate never released")
            proc = _real_popen(*args, **kwargs)
            self.created.append(proc)
            return proc

        with mock.patch.object(appmod.subprocess, "Popen", gated_popen):
            start_result = {}
            starter = threading.Thread(
                target=lambda: start_result.update(self.client.post(
                    "/api/run", json={"code": SIMPLE_CODE}).get_json())
            )
            starter.start()
            self.assertTrue(entered.wait(10), "starter never reached Popen")

            # The run is registered (CURRENT exists) but its child does not
            # exist yet: terminate must be remembered, not lost.
            with appmod.COND:
                old_id = appmod.CURRENT.run_id
                self.assertIsNone(appmod.CURRENT.proc)
                self.assertFalse(appmod.CURRENT.proc_started)

            terminate_result = {}

            def do_terminate():
                res = self.client.post("/api/terminate", json={"run_id": old_id})
                terminate_result["status_code"] = res.status_code
                terminate_result["body"] = res.get_json()

            terminator = threading.Thread(target=do_terminate)
            terminator.start()
            # Terminate is now parked waiting for startup; it must not
            # succeed before the child handle exists and is reaped.
            terminator.join(timeout=0.5)
            self.assertTrue(terminator.is_alive(),
                            "terminate returned while Popen was still gated")

            release.set()
            terminator.join(timeout=10)
            starter.join(timeout=10)
            self.assertFalse(starter.is_alive())
            self.assertFalse(terminator.is_alive())

        self.assertEqual(terminate_result["status_code"], 200, terminate_result)
        self.assertEqual(terminate_result["body"]["status"], "terminated")
        self.assertEqual(start_result["run_id"], old_id)

        with appmod.COND:
            old_run = appmod.CURRENT
            old_proc = old_run.proc
            self.assertEqual(old_run.status, "terminated")
            self.assertTrue(old_run.reaped, "old child not reaped after terminate")
        self.assert_proc_dead(old_proc)

        # Immediate re-run: the old child must be fully reaped first, and
        # only one run may exist afterwards.
        n_before = len(self.created)
        new_id = self.start_run()
        self.assertNotEqual(new_id, old_id)
        self.assertEqual(len(self.created), n_before + 1)
        new_proc = self.created[-1]
        self.assertIsNone(new_proc.poll(), "new child should be alive")
        self.assert_proc_dead(old_proc)  # old child still dead, not reused

        # The stale identifier can no longer drive (terminate) the old run.
        stale = self.client.post("/api/terminate", json={"run_id": old_id})
        self.assertEqual(stale.status_code, 409)

        self.wait_paused(new_id)
        self.cleanup_run(new_id)

    # ---- 2. immediate re-run after an ordinary terminate ----------------

    def test_immediate_rerun_after_terminate(self):
        old_id = self.start_run()
        self.wait_paused(old_id)
        old_proc = appmod.CURRENT.proc

        res = self.client.post("/api/terminate", json={"run_id": old_id})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "terminated")
        self.assert_proc_dead(old_proc)
        self.assertTrue(appmod.CURRENT.reaped)

        new_id = self.start_run()  # no delay: must not 409
        self.assertNotEqual(new_id, old_id)
        self.wait_paused(new_id)
        self.assertEqual(self.state()["status"], "paused")
        self.cleanup_run(new_id)

    # ---- 3. late events must not resurrect a terminal run ---------------

    def test_apply_event_ignores_late_and_foreign_events(self):
        old = appmod.Run("x = 1\n")
        current = appmod.Run("y = 2\n")
        pause = {"type": "pause", "line": 1, "locals": {"a": "1"}}

        with appmod.COND:
            # Terminated run: buffered pause/end/stdout events are dropped.
            appmod.CURRENT = old
            old.status = "terminated"
            appmod._apply_event(old, dict(pause))
            appmod._apply_event(old, {"type": "end"})
            appmod._apply_event(old, {"type": "stdout", "data": "late"})
            self.assertEqual(old.status, "terminated")
            self.assertEqual(old.pause_seq, 0)
            self.assertIsNone(old.current_line)
            self.assertEqual(old.stdout, "")

            # Finished run cannot go back to paused either.
            old.status = "finished"
            appmod._apply_event(old, dict(pause))
            self.assertEqual(old.status, "finished")

            # Events belonging to a run that is no longer current are
            # dropped even if that run object still looks active.
            appmod.CURRENT = current
            current.status = "running"
            old.status = "running"
            appmod._apply_event(old, dict(pause))
            self.assertEqual(old.status, "running")
            self.assertEqual(old.pause_seq, 0)
            appmod.CURRENT = None  # neither synthetic run owns a process

    def test_late_events_after_terminate_end_to_end(self):
        # Child floods stdout events; after continue the pipe is full of
        # events not yet read when terminate lands. Draining them after the
        # kill must not change the terminated state.
        run_id = self.start_run(FLOOD_CODE)
        self.wait_paused(run_id)
        seq = self.state()["pause_seq"]

        res = self.client.post("/api/continue", json={"run_id": run_id, "pause_seq": seq})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(wait_for(lambda: self.state()["status"] == "running"))
        # Let the child get ahead and stuff the pipe, then kill it.
        time.sleep(0.2)
        proc = appmod.CURRENT.proc
        res = self.client.post("/api/terminate", json={"run_id": run_id})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "terminated")

        # terminate succeeded => child reaped => reader finished draining;
        # even so, keep polling: nothing may flip the state back.
        self.assertTrue(appmod.CURRENT.reaped)
        self.assert_proc_dead(proc)
        for _ in range(30):
            snap = self.state()
            self.assertEqual(snap["run_id"], run_id)
            self.assertEqual(snap["status"], "terminated")
            time.sleep(0.01)

    # ---- 4. normal pause / step / continue semantics --------------------

    def test_normal_pause_step_continue_and_dedup(self):
        run_id = self.start_run("x = 1\nx = 2\nx = 3\n")
        self.wait_paused(run_id)

        snap = self.state()
        self.assertEqual(snap["current_line"], 1)
        self.assertEqual(snap["pause_seq"], 1)

        ok = self.client.post("/api/step", json={"run_id": run_id, "pause_seq": 1})
        self.assertEqual(ok.status_code, 200)
        self.assertTrue(wait_for(lambda: self.state()["status"] == "paused"))
        self.assertEqual(self.state()["current_line"], 2)
        self.assertEqual(self.state()["pause_seq"], 2)

        # Duplicate of the consumed pause and stale seq are 409, no advance.
        dup = self.client.post("/api/step", json={"run_id": run_id, "pause_seq": 1})
        self.assertEqual(dup.status_code, 409)
        stale = self.client.post("/api/continue",
                                 json={"run_id": run_id, "pause_seq": 1})
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(self.state()["current_line"], 2)

        cont = self.client.post("/api/continue",
                                json={"run_id": run_id, "pause_seq": 2})
        self.assertEqual(cont.status_code, 200)
        self.assertTrue(wait_for(lambda: self.state()["status"] == "finished"))
        self.assertTrue(appmod.CURRENT.reaped)

        # A terminate for an already-finished run is stale -> 409, never 500.
        late = self.client.post("/api/terminate", json={"run_id": run_id})
        self.assertEqual(late.status_code, 409)

    # ---- 5. runaway infinite loop termination ---------------------------

    def test_infinite_loop_terminate(self):
        run_id = self.start_run(LOOP_CODE)
        self.wait_paused(run_id)
        seq = self.state()["pause_seq"]
        res = self.client.post("/api/continue",
                               json={"run_id": run_id, "pause_seq": seq})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(wait_for(lambda: self.state()["status"] == "running"))
        time.sleep(0.1)  # burning CPU in the child now

        proc = appmod.CURRENT.proc
        res = self.client.post("/api/terminate", json={"run_id": run_id})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "terminated")
        self.assert_proc_dead(proc)
        self.assertTrue(appmod.CURRENT.reaped)

        # A fresh run is accepted right away.
        new_id = self.start_run()
        self.assertNotEqual(new_id, run_id)
        self.cleanup_run(new_id)

    # ---- 6. conflict rules stay 409 -------------------------------------

    def test_single_run_conflict_and_expired_controls(self):
        first_id = self.start_run()
        self.wait_paused(first_id)

        # A second concurrent run conflicts with 409.
        conflict = self.client.post("/api/run", json={"code": SIMPLE_CODE})
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(len(self.created), 1, "conflicting run spawned a process")

        # Step with an unknown run id is 409.
        bad = self.client.post("/api/step",
                               json={"run_id": "deadbeef", "pause_seq": 1})
        self.assertEqual(bad.status_code, 409)

        # Terminate the first run, then start a new one: the old run id is
        # expired for every control endpoint.
        self.assertEqual(
            self.client.post("/api/terminate", json={"run_id": first_id}).status_code,
            200)
        second_id = self.start_run()
        for path in ("/api/terminate",):
            res = self.client.post(path, json={"run_id": first_id})
            self.assertEqual(res.status_code, 409)
        for path in ("/api/step", "/api/continue"):
            res = self.client.post(path, json={"run_id": first_id, "pause_seq": 1})
            self.assertEqual(res.status_code, 409)
        self.wait_paused(second_id)
        self.cleanup_run(second_id)

    def cleanup_run(self, run_id):
        """Terminate a paused/active run and assert the child is reaped."""
        proc = appmod.CURRENT.proc
        res = self.client.post("/api/terminate", json={"run_id": run_id})
        self.assertEqual(res.status_code, 200, res.get_data(as_text=True))
        self.assert_proc_dead(proc)

    # ---- 7. uncontrolled interleaving stress ---------------------------

    def test_stress_run_terminate_rerun(self):
        n_threads = 4
        cycles = 40
        errors = []
        max_alive = {"v": 0}
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                alive = sum(1 for p in self.created if p.poll() is None)
                if alive > max_alive["v"]:
                    max_alive["v"] = alive
                time.sleep(0.002)

        monitor = threading.Thread(target=sampler)
        monitor.start()

        def worker(tid):
            client = appmod.app.test_client()
            try:
                for i in range(cycles):
                    res = client.post("/api/run", json={"code": SIMPLE_CODE})
                    if res.status_code == 409:
                        continue  # another thread holds the single run slot
                    self.assertEqual(res.status_code, 200)
                    rid = res.get_json()["run_id"]
                    # Alternate: terminate straight away (may still be in
                    # startup) or after the first pause.
                    if (i + tid) % 2 == 0:
                        time.sleep(0.002)
                    else:
                        if not wait_for(
                                lambda c=client, r=rid:
                                c.get("/api/state").get_json().get("run_id") == r
                                and c.get("/api/state").get_json()["status"] == "paused",
                                timeout=10):
                            errors.append(f"{tid}/{i}: never paused")
                            continue
                    t = client.post("/api/terminate", json={"run_id": rid})
                    if t.status_code != 200 or t.get_json()["status"] != "terminated":
                        errors.append(f"{tid}/{i}: terminate {t.status_code} {t.get_json()}")
                        continue
                    # Immediate re-run: must be admitted once reaped and
                    # then terminated as well.
                    r2 = client.post("/api/run", json={"code": SIMPLE_CODE})
                    if r2.status_code == 200:
                        rid2 = r2.get_json()["run_id"]
                        t2 = client.post("/api/terminate", json={"run_id": rid2})
                        if t2.status_code != 200:
                            errors.append(f"{tid}/{i}: rerun terminate {t2.status_code}")
            except Exception as exc:  # noqa: BLE001 - report from worker
                errors.append(f"worker {tid}: {exc!r}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            self.assertFalse(t.is_alive(), "worker hung (deadlock?)")
        stop.set()
        monitor.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(self.created), 4)
        self.assertLessEqual(max_alive["v"], 1,
                             f"up to {max_alive['v']} children lived at once")
        # Every child ever created must be dead AND reaped (poll not None),
        # with no survivors left behind.
        for proc in self.created:
            self.assertIsNotNone(proc.poll(), "child never reaped")
            with self.assertRaises(ProcessLookupError):
                os.kill(proc.pid, 0)


    # ---- 8. Popen failure racing terminate ------------------------------

    def test_popen_failure_during_pending_terminate(self):
        entered = threading.Event()
        terminator_started = threading.Event()
        release = threading.Event()

        def failing_popen(*args, **kwargs):
            entered.set()
            # Only fail after terminate has parked waiting for startup.
            self.assertTrue(terminator_started.wait(timeout=10))
            self.assertTrue(release.wait(timeout=10))
            raise OSError("spawn failed")

        with mock.patch.object(appmod.subprocess, "Popen", failing_popen):
            result = {}

            def do_run():
                res = self.client.post("/api/run", json={"code": SIMPLE_CODE})
                result["run"] = (res.status_code, res.get_json())

            t = threading.Thread(target=do_run)
            t.start()
            self.assertTrue(entered.wait(10))
            with appmod.COND:
                rid = appmod.CURRENT.run_id

            def do_terminate():
                terminator_started.set()
                res = self.client.post("/api/terminate", json={"run_id": rid})
                result["term"] = (res.status_code, res.get_json())

            tt = threading.Thread(target=do_terminate)
            tt.start()
            release.set()  # let Popen fail while terminate is parked
            t.join(timeout=10)
            tt.join(timeout=10)
            self.assertFalse(t.is_alive())
            self.assertFalse(tt.is_alive())
        self.assertEqual(result["run"][0], 200)
        self.assertEqual(result["term"][0], 200)
        with appmod.COND:
            self.assertIn(appmod.CURRENT.status, appmod.TERMINAL_STATES)
            self.assertTrue(appmod.CURRENT.reaped)

        # Next run admitted normally.
        new_id = self.start_run()
        self.wait_paused(new_id)
        self.cleanup_run(new_id)

    # ---- 9. multiple waiters parked on an un-reaped terminal run ---------

    @contextlib.contextmanager
    def gated_popen(self, max_gated, before_spawn=None):
        """Real Popen, but the first ``max_gated`` children get a wait()
        that parks on a test-released gate: the child exits on its own yet
        stays un-reaped (zombie) until the test releases the gate.  This
        reproduces "run reported ended, exit cleanup not done yet" with a
        real process and a fully controlled interleaving.  Yields the list
        of per-child release events, in creation order."""
        real_popen = self._real_popen
        gates = []

        class GatedProc:
            def __init__(self, proc):
                self._proc = proc
                self.gate = threading.Event()
                gates.append(self.gate)

            def wait(self, timeout=None):
                self.gate.wait(10)  # bounded park; tests release the gate
                return self._proc.wait(timeout=timeout)

            def __getattr__(self, name):
                return getattr(self._proc, name)

        def wrapping_popen(*args, **kwargs):
            if before_spawn is not None:
                before_spawn()
            proc = real_popen(*args, **kwargs)
            self.created.append(proc)
            if len(gates) < max_gated:
                return GatedProc(proc)
            return proc

        with mock.patch.object(appmod.subprocess, "Popen", wrapping_popen):
            yield gates
        for gate in gates:
            gate.set()  # never leave a reader thread parked after the test

    def terminal_unreaped_run(self, code=SIMPLE_CODE):
        """Start a real run and drive it to natural end while its reaping is
        parked on a gate: terminal status, child exited, not yet reaped.
        Returns (run_id, Run)."""
        run_id = self.start_run(code)
        self.wait_paused(run_id)
        seq = self.state()["pause_seq"]
        res = self.client.post("/api/continue",
                               json={"run_id": run_id, "pause_seq": seq})
        self.assertEqual(res.status_code, 200)
        with appmod.COND:
            run = appmod.CURRENT
        ok = wait_for(lambda: run.status == "finished" and not run.reaped
                      and run.proc.poll() is not None)
        self.assertTrue(ok, "run never reached terminal-but-unreaped state")
        return run_id, run

    def test_woken_waiter_rechecked_against_current_run(self):
        """Two starters wait for an un-reaped terminal run; after the old
        child is reaped, the first admitted run reports ended but its exit
        cleanup is still pending.  The second waiter must not register or
        spawn the next run until that replacement run is reaped too."""
        admitted = []  # runs that were CURRENT at each spawn, in order
        spawn_errors = []

        def note_spawn():
            with appmod.COND:
                for prev in admitted:
                    if not prev.reaped:
                        spawn_errors.append(
                            f"new child spawned while run {prev.run_id} unreaped")
                admitted.append(appmod.CURRENT)

        with self.gated_popen(max_gated=2, before_spawn=note_spawn) as gates:
            old_id, old_run = self.terminal_unreaped_run()
            self.assertEqual(len(gates), 1)

            # Two starters park on the old run's reaping at the same time.
            real_wait = appmod._wait_reaped
            waits = []  # (thread, run) each starter is parked on
            both_parked = threading.Event()
            b_may_recheck = threading.Event()
            threads = {}

            def recording_wait(run, timeout=None):
                me = threading.current_thread()
                waits.append((me, run))
                if len({t for t, _ in waits}) == 2:
                    both_parked.set()
                ok = real_wait(run, timeout=timeout if timeout is not None
                               else appmod.REAP_WAIT_TIMEOUT)
                if me is threads["b"] and run is old_run:
                    # Hold B's re-check (without holding COND) until the
                    # replacement run is terminal-but-unreaped, so the
                    # wake-up interleaving is deterministic.
                    appmod.COND.release()
                    self.assertTrue(b_may_recheck.wait(timeout=10))
                    appmod.COND.acquire()
                return ok

            results = {}

            def starter(name, code):
                res = self.client.post("/api/run", json={"code": code})
                results[name] = (res.status_code, res.get_json())

            with mock.patch.object(appmod, "_wait_reaped", recording_wait):
                threads["a"] = threading.Thread(target=starter,
                                                args=("a", "a = 1\n"))
                threads["b"] = threading.Thread(target=starter,
                                                args=("b", "b = 2\n"))
                threads["a"].start()
                threads["b"].start()
                self.assertTrue(both_parked.wait(10), "starters never parked")

                # Old child reaped: A is admitted, B stays held for now.
                gates[0].set()
                self.assertTrue(wait_for(lambda: "a" in results))
                self.assertEqual(results["a"][0], 200, results)
                run_a_id = results["a"][1]["run_id"]
                self.assertNotEqual(run_a_id, old_id)

                # The replacement run now reports ended, but its child is
                # exited-yet-unreaped (cleanup still pending).
                self.wait_paused(run_a_id)
                seq = self.state()["pause_seq"]
                res = self.client.post("/api/continue",
                                       json={"run_id": run_a_id,
                                             "pause_seq": seq})
                self.assertEqual(res.status_code, 200)
                with appmod.COND:
                    run_a = appmod.CURRENT
                self.assertTrue(wait_for(
                    lambda: run_a.status == "finished" and not run_a.reaped
                    and self.created[1].poll() is not None),
                    "replacement run never reached terminal-but-unreaped")
                self.assertEqual(len(gates), 2)

                # B wakes from the old run's reaping now.  It must wait for
                # the replacement run's reaping instead of leapfrogging it.
                b_may_recheck.set()
                self.assertTrue(wait_for(
                    lambda: any(t is threads["b"] and r is run_a
                                for t, r in waits), timeout=5),
                    "woken starter did not wait for the replacement's reaping")
                self.assertNotIn("b", results)  # still parked, not admitted
                # The previous identity stays valid until its run is reaped.
                self.assertEqual(self.state()["run_id"], run_a_id)

                gates[1].set()  # replacement reaped: B may be admitted now
                threads["b"].join(timeout=10)
                threads["a"].join(timeout=10)
                self.assertFalse(threads["b"].is_alive())

            self.assertEqual(results["b"][0], 200, results)
            run_b_id = results["b"][1]["run_id"]
            self.assertNotEqual(run_b_id, run_a_id)
            self.assertEqual(spawn_errors, [])
            self.assertEqual(len(self.created), 3)
            with appmod.COND:
                self.assertTrue(old_run.reaped)
                self.assertTrue(run_a.reaped)
                self.assertEqual(appmod.CURRENT.run_id, run_b_id)
            self.assertEqual(self.state()["run_id"], run_b_id)
            self.wait_paused(run_b_id)
            self.cleanup_run(run_b_id)

    def test_woken_waiter_conflicts_with_active_replacement(self):
        """Two starters wait for an un-reaped terminal run; once the old
        child is reaped exactly one is admitted, and the other sees the
        active replacement holding the slot and gets 409 (no new process)."""
        with self.gated_popen(max_gated=1) as gates:
            old_id, old_run = self.terminal_unreaped_run()

            real_wait = appmod._wait_reaped
            parked = []
            both_parked = threading.Event()

            def recording_wait(run, timeout=None):
                parked.append(threading.current_thread())
                if len(set(parked)) == 2:
                    both_parked.set()
                return real_wait(run, timeout=timeout if timeout is not None
                                 else appmod.REAP_WAIT_TIMEOUT)

            results = {}

            def starter(name):
                res = self.client.post("/api/run", json={"code": SIMPLE_CODE})
                results[name] = (res.status_code, res.get_json())

            with mock.patch.object(appmod, "_wait_reaped", recording_wait):
                threads = [threading.Thread(target=starter, args=(n,))
                           for n in ("a", "b")]
                for t in threads:
                    t.start()
                self.assertTrue(both_parked.wait(10), "starters never parked")
                gates[0].set()  # old child reaped: exactly one may proceed
                for t in threads:
                    t.join(timeout=10)
                    self.assertFalse(t.is_alive())

            codes = sorted(r[0] for r in results.values())
            self.assertEqual(codes, [200, 409], results)
            self.assertEqual(len(self.created), 2,
                             "conflicting waiter spawned a process")
            winner = next(r for r in results.values() if r[0] == 200)
            winner_id = winner[1]["run_id"]
            self.assertNotEqual(winner_id, old_id)
            self.assertEqual(self.state()["run_id"], winner_id)
            self.wait_paused(winner_id)
            self.cleanup_run(winner_id)

    def test_wait_for_reap_timeout_keeps_current_run(self):
        """A starter waiting for the previous child to be reaped times out
        with 409 and must not replace the current run early; once reaping
        completes, the re-run is admitted."""
        with self.gated_popen(max_gated=1) as gates:
            old_id, old_run = self.terminal_unreaped_run()
            with mock.patch.object(appmod, "REAP_WAIT_TIMEOUT", 0.3):
                started = time.monotonic()
                res = self.client.post("/api/run", json={"code": SIMPLE_CODE})
                elapsed = time.monotonic() - started
            self.assertEqual(res.status_code, 409)
            self.assertGreaterEqual(elapsed, 0.2,
                                    "409 returned without actually waiting")
            self.assertLess(elapsed, 5)
            # The timed-out request changed nothing: same current run, still
            # un-reaped, and no new child was spawned.
            self.assertEqual(self.state()["run_id"], old_id)
            self.assertFalse(old_run.reaped)
            self.assertEqual(len(self.created), 1)

            gates[0].set()  # reaping completes -> re-run is admitted
            self.assertTrue(wait_for(lambda: old_run.reaped))
            new_id = self.start_run()
            self.assertNotEqual(new_id, old_id)
            self.wait_paused(new_id)
            self.cleanup_run(new_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
