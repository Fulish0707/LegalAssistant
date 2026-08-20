import hashlib
import os
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timezone


class DailyQuotaTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_file = os.path.join(self.tmpdir.name, "quota.db")
        self._create_schema()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_schema(self):
        with sqlite3.connect(self.db_file) as connection:
            connection.execute(
                """
                CREATE TABLE daily_ai_quota (
                    identity_kind TEXT NOT NULL,
                    identity_value TEXT NOT NULL,
                    quota_day TEXT NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (identity_kind, identity_value, quota_day)
                )
                """
            )

    def _connection(self):
        return sqlite3.connect(self.db_file, timeout=5)

    def test_allows_twenty_reservations_then_rejects_the_twenty_first(self):
        from quota import reserve_daily_quota

        with self._connection() as connection:
            results = [
                reserve_daily_quota(connection, "user", "account-1", now=datetime(2026, 8, 20, tzinfo=timezone.utc))
                for _ in range(21)
            ]

        self.assertTrue(all(result.allowed for result in results[:20]))
        self.assertEqual(results[19].used, 20)
        self.assertEqual(results[19].remaining, 0)
        self.assertFalse(results[20].allowed)
        self.assertEqual(results[20].used, 20)

    def test_identity_and_utc_day_are_isolated(self):
        from quota import reserve_daily_quota

        with self._connection() as connection:
            for _ in range(20):
                reserve_daily_quota(connection, "user", "account-1", now=datetime(2026, 8, 20, tzinfo=timezone.utc))
            other_user = reserve_daily_quota(connection, "user", "account-2", now=datetime(2026, 8, 20, tzinfo=timezone.utc))
            next_day = reserve_daily_quota(connection, "user", "account-1", now=datetime(2026, 8, 21, tzinfo=timezone.utc))

        self.assertTrue(other_user.allowed)
        self.assertTrue(next_day.allowed)
        self.assertEqual(other_user.used, 1)
        self.assertEqual(next_day.used, 1)

    def test_concurrent_reservations_never_exceed_limit(self):
        from quota import reserve_daily_quota

        start = threading.Barrier(31)
        results = []
        result_lock = threading.Lock()

        def reserve():
            connection = self._connection()
            try:
                start.wait()
                result = reserve_daily_quota(connection, "guest_ip", "hashed-ip", now=datetime(2026, 8, 20, tzinfo=timezone.utc))
                with result_lock:
                    results.append(result)
            finally:
                connection.close()

        threads = [threading.Thread(target=reserve) for _ in range(30)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 30)
        self.assertEqual(sum(result.allowed for result in results), 20)
        self.assertTrue(all(result.used <= 20 for result in results))

    def test_guest_identity_hashes_normalized_ip_without_retaining_raw_ip(self):
        from quota import guest_ip_identity

        identity = guest_ip_identity(" 2001:0db8::1 ", "test-salt")

        self.assertEqual(identity, hashlib.sha256(b"test-salt:2001:db8::1").hexdigest())
        self.assertNotIn("2001:0db8::1", identity)

    def test_client_ip_uses_forwarded_chain_only_for_configured_trusted_proxies(self):
        from quota import client_ip_from_request

        class Request:
            remote_addr = "10.0.0.8"
            access_route = ("198.51.100.14", "10.0.0.7", "10.0.0.8")

        self.assertEqual(client_ip_from_request(Request(), trusted_proxy_count=0), "10.0.0.8")
        self.assertEqual(client_ip_from_request(Request(), trusted_proxy_count=2), "198.51.100.14")
