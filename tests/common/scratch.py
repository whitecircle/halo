"""The spelling shared by test scratch dirs and the sweep; import-free, since the launcher session must not import
torch or ``halo``."""

# The root conftest sweeps leaked dirs by this spelling alone, so it never matches another program's dirs in a shared
# TMPDIR.
SCRATCH_DIR_TAG = "halo-test-"
