import os as _os

# On a weights-free runner (the CI "fast" tier), these modules must not even be IMPORTED:
# they import detector, which loads ~380MB of weights at import. pytest imports every module
# at collection time, so a marker deselection is not enough — we skip collection entirely.
collect_ignore = []
if _os.environ.get("VOICEGUARD_CI_FAST"):
    collect_ignore = ["test_api.py", "test_detector.py", "test_worker.py",
                      "test_golden.py", "test_gradcam.py", "test_submodel_health.py",
                      "test_eval_bundle_weights.py"]

import io
import os
import pytest


class FakeS3:
    """In-memory stand-in for a boto3 S3 client — only the methods remote_store uses.
    Keys are (bucket, key) tuples -> bytes."""

    def __init__(self):
        self.store = {}

    def upload_file(self, filename, bucket, key):
        with open(filename, "rb") as f:
            self.store[(bucket, key)] = f.read()

    def download_file(self, bucket, key, filename):
        data = self.store[(bucket, key)]                 # KeyError if absent
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "wb") as f:
            f.write(data)

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.store:
            raise KeyError((Bucket, Key))
        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}


@pytest.fixture
def fake_s3():
    return FakeS3()
