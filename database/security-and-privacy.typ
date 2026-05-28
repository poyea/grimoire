= Database Security and Privacy

Database security must be designed in — SQL injection, privilege escalation, and data leakage are the most common causes of data breaches. Privacy-preserving computation (differential privacy, encryption in use) adds a layer beyond access control.

*See also:* _Database Foundations_, _Observability and Self-Driving Databases_

== SQL Injection

*SQL injection* is the most prevalent web vulnerability (OWASP #1 for decades). User input is embedded in a SQL string and interpreted as SQL syntax.

```python
# VULNERABLE: string interpolation
def get_user_UNSAFE(username: str) -> dict | None:
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return db.execute(query).fetchone()

# Attack: username = "' OR '1'='1"
# Resulting query: SELECT * FROM users WHERE username = '' OR '1'='1'
# → returns ALL users

# Attack: username = "'; DROP TABLE users; --"
# → deletes the users table

# SAFE: parameterized query (prepared statement)
def get_user_SAFE(username: str) -> dict | None:
    query = "SELECT * FROM users WHERE username = %s"
    return db.execute(query, (username,)).fetchone()

# username = "'; DROP TABLE users; --"  → treated as literal string, not SQL
```

*Parameterized queries are the only reliable defense.* ORMs use them by default:

```python
# SQLAlchemy (parameterized automatically)
user = session.query(User).filter(User.username == username).first()

# Django ORM
user = User.objects.get(username=username)

# Safe raw SQL in Django:
from django.db import connection
with connection.cursor() as cur:
    cur.execute("SELECT * FROM users WHERE username = %s", [username])
```

*Second-order injection:* data stored safely can be re-read and used in a dynamic query without parameterization:

```python
# Stored username: attacker saves "admin'--" to database
# Later, someone does:
username = fetch_username_from_db(user_id)   # looks "safe"
query = f"SELECT * FROM profiles WHERE name = '{username}'"  # INJECTION!
# Always parameterize, even with "trusted" database values
```

== Access Control

=== Discretionary Access Control (DAC)

SQL GRANT/REVOKE controls who can access what.

```sql
-- Grant specific privileges
GRANT SELECT, INSERT ON orders TO app_user;
GRANT SELECT ON customer_summary TO readonly_user;
GRANT EXECUTE ON FUNCTION calculate_tax TO app_user;

-- Role-based: grant role, not individual privileges
CREATE ROLE analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA reporting TO analyst;
GRANT analyst TO alice;
GRANT analyst TO bob;

-- Revoke
REVOKE INSERT ON orders FROM app_user;
REVOKE analyst FROM alice;

-- Column-level security (limit which columns are readable)
GRANT SELECT (customer_id, order_date, amount) ON orders TO analyst;
-- analyst cannot SELECT the 'notes' column
```

=== Row-Level Security (RLS)

Enforce data isolation at the row level — essential for multi-tenant systems.

```sql
-- Enable RLS on orders table
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

-- Policy: users can only see their own orders
CREATE POLICY user_isolation ON orders
    USING (customer_id = current_setting('app.current_user_id')::bigint);

-- Policy: admin role sees all
CREATE POLICY admin_access ON orders
    TO admin_role
    USING (true);

-- Application: set current user at connection time
SET app.current_user_id = 42;
SELECT * FROM orders;   -- only returns rows where customer_id = 42
```

*Implementation:* PostgreSQL appends the RLS predicate to every query on that table at the planner level. The application cannot bypass it via SQL.

```sql
-- Multi-tenant SaaS: isolate by tenant_id
ALTER TABLE data ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON data
    USING (tenant_id = current_setting('app.tenant_id')::int);

-- At connection time (or per-transaction):
SET LOCAL app.tenant_id = 7;
-- Now ALL queries see only tenant 7's data, even if app forgets WHERE clause
```

== Encryption

=== Encryption at Rest

```bash
# PostgreSQL: filesystem-level encryption (OS handles this)
# For column-level: use pgcrypto extension

# Encrypt sensitive column at write time
INSERT INTO users (name, ssn_encrypted)
VALUES ('Alice', pgp_sym_encrypt('123-45-6789', 'encryption_key'));

# Decrypt at read time
SELECT name, pgp_sym_decrypt(ssn_encrypted, 'encryption_key') AS ssn
FROM users;
```

*Key management:* never store the encryption key in the database. Use HashiCorp Vault, AWS KMS, or GCP CMEK. The database stores encrypted ciphertext; the key lives in the external KMS.

=== Encryption in Transit

Always use TLS for client-server connections and replication. See _TLS_ (Networking volume) for protocol details.

```sql
-- Force SSL in PostgreSQL (postgresql.conf)
-- ssl = on
-- ssl_cert_file = 'server.crt'
-- ssl_key_file = 'server.key'

-- Per-user: require SSL for a specific role
ALTER ROLE app_user CONNECTION LIMIT 100;
-- In pg_hba.conf: hostssl all app_user 0.0.0.0/0 scram-sha-256
```

=== Transparent Data Encryption (TDE)

InnoDB TDE encrypts tablespace files on disk; decrypts in memory. Key stored in a keyring plugin.

```sql
-- MySQL InnoDB TDE
-- my.cnf: early-plugin-load=keyring_file.so
--         keyring_file_data=/var/lib/mysql-keyring/keyring

-- Encrypt a table
ALTER TABLE orders ENCRYPTION='Y';

-- Verify
SELECT NAME, ENCRYPTION FROM information_schema.INNODB_TABLESPACES
WHERE NAME LIKE '%orders%';
```

== Authentication

```sql
-- PostgreSQL: SCRAM-SHA-256 (strongest built-in)
-- In pg_hba.conf:
-- host all all 0.0.0.0/0 scram-sha-256

-- Create user with strong password
CREATE USER app_user WITH PASSWORD 'use-vault-not-this';

-- Password hashing: PostgreSQL stores SCRAM-SHA-256 verifier (not plaintext)
SELECT rolpassword FROM pg_authid WHERE rolname = 'app_user';
-- SCRAM-SHA-256$4096:...  (iterated hash + salt)
```

*Never hardcode credentials in application code.* Use environment variables or secret managers:

```python
import os
import psycopg2

conn = psycopg2.connect(
    host     = os.environ["PGHOST"],
    dbname   = os.environ["PGDATABASE"],
    user     = os.environ["PGUSER"],
    password = os.environ["PGPASSWORD"],   # from secret manager at startup
    sslmode  = "require",
)
```

== Differential Privacy

Differential privacy (Dwork et al. 2006) provides a mathematical guarantee that the inclusion or exclusion of a single person's data changes the query result by at most a bounded amount.

*Definition:* a randomized mechanism $M$ satisfies $(epsilon, delta)$-differential privacy if for all neighboring datasets $D$ and $D'$ (differing in one row) and all outputs $S$:

$Pr[M(D) in S] <= e^epsilon dot Pr[M(D') in S] + delta$

*Laplace mechanism* for counting queries:

```python
import numpy as np

def dp_count(values: list, epsilon: float, sensitivity: int = 1) -> float:
    """
    Return differentially private COUNT with privacy parameter epsilon.
    sensitivity = 1 (adding/removing one row changes COUNT by at most 1).
    """
    true_count = len(values)
    noise_scale = sensitivity / epsilon   # Laplace distribution parameter
    noise = np.random.laplace(0, noise_scale)
    return true_count + noise

# epsilon = 1.0: moderate privacy. epsilon << 1: strong privacy, high noise.
dp_count(users, epsilon=1.0)

# Gaussian mechanism for (epsilon, delta)-DP
def dp_count_gaussian(values, epsilon, delta, sensitivity=1):
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    return len(values) + np.random.normal(0, sigma)
```

*Google RAPPOR, Apple DP:* differential privacy is deployed in telemetry collection — raw user data never leaves the device; only noisy aggregate statistics are reported.

```python
# Differentially private mean (clip then add noise)
def dp_mean(values: list[float], epsilon: float,
            clip_lo: float, clip_hi: float) -> float:
    n = len(values)
    # Clip to bound sensitivity
    clipped = [max(clip_lo, min(clip_hi, v)) for v in values]
    sensitivity = (clip_hi - clip_lo) / n    # sensitivity of mean query
    true_sum = sum(clipped)
    noisy_sum = true_sum + np.random.laplace(0, sensitivity * n / epsilon)
    return noisy_sum / n
```

== Audit Logging

```sql
-- PostgreSQL: log all DDL and specific DML
-- postgresql.conf:
-- log_statement = 'ddl'        -- log CREATE/DROP/ALTER
-- log_statement = 'mod'        -- log INSERT/UPDATE/DELETE/TRUNCATE
-- log_statement = 'all'        -- log everything (high volume)
-- log_connections = on
-- log_disconnections = on

-- pgaudit extension: structured audit log
-- shared_preload_libraries = 'pgaudit'
-- pgaudit.log = 'ddl, write, role'

-- Query the audit log
SELECT session_user, command_tag, object_type, object_name, application_name, client_addr
FROM pgaudit.log_view
WHERE command_tag = 'DROP TABLE'
ORDER BY log_time DESC;
```

== Common Vulnerabilities Summary

#table(
  columns: (auto, auto),
  [*Vulnerability*], [*Mitigation*],
  [SQL injection],            [Parameterized queries; ORMs; input validation],
  [Privilege escalation],     [Principle of least privilege; RLS; separate DB users per service],
  [Credential exposure],      [Secrets manager; environment vars; never in code/logs],
  [Unencrypted data at rest], [TDE; filesystem encryption; column-level pgcrypto],
  [Unencrypted in transit],   [Require TLS (sslmode=require); certificate pinning],
  [Mass data extraction],     [Rate limiting; row-count limits; query logging; DLP tools],
  [Backup exposure],          [Encrypt backups; restrict S3 bucket access; audit access logs],
)

== References

Dwork, C., McSherry, F., Nissim, K., Smith, A. (2006). "Calibrating Noise to Sensitivity in Private Data Analysis." TCC.

OWASP. "SQL Injection Prevention Cheat Sheet." https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html

PostgreSQL Documentation. "Row Security Policies." https://www.postgresql.org/docs/current/ddl-rowsecurity.html

Erlingsson, Ú., Pihur, V., Korolova, A. (2014). "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response." CCS. (Google DP telemetry)

Dwork, C., Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy." Foundations and Trends in TCS.
