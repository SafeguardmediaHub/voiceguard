"""remote_store.py — DigitalOcean Spaces (S3-compatible) backend for model bundles.

The ONLY module that imports boto3, and it does so lazily inside make_client so the
module imports fine without boto3 (and tests run against an injected fake client).
Config comes from SPACES_* env vars; see docs/CI-and-model-store.md.
"""
import os

_REQUIRED = ("SPACES_KEY", "SPACES_SECRET", "SPACES_ENDPOINT", "SPACES_REGION", "SPACES_BUCKET")


def make_client(env=os.environ):
    """boto3 S3 client for DigitalOcean Spaces from SPACES_* env vars.
    Raises RuntimeError naming the first missing required var."""
    for name in _REQUIRED:
        if not env.get(name):
            raise RuntimeError(f"missing required env var: {name}")
    import boto3                                          # lazy: only needed for a real client
    return boto3.client(
        "s3",
        region_name=env["SPACES_REGION"],
        endpoint_url=env["SPACES_ENDPOINT"],
        aws_access_key_id=env["SPACES_KEY"],
        aws_secret_access_key=env["SPACES_SECRET"],
    )


def bucket_prefix(env=os.environ):
    """(bucket, prefix) from env; prefix default 'voiceguard/model_store', no trailing slash."""
    return env["SPACES_BUCKET"], env.get("SPACES_PREFIX", "voiceguard/model_store").rstrip("/")


def upload_file(client, bucket, key, local_path):
    client.upload_file(local_path, bucket, key)


def download_file(client, bucket, key, local_path):
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    client.download_file(bucket, key, local_path)


def download_bytes(client, bucket, key):
    """Object bytes, or None if the key is absent (404). Other errors propagate."""
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()
    except Exception as e:                                # noqa: BLE001 — normalized below
        if _is_not_found(e):
            return None
        raise


def _is_not_found(e):
    # Real boto3 raises ClientError with response['Error']['Code'] in {NoSuchKey,404,NotFound};
    # the in-memory FakeS3 raises KeyError. Treat both as "absent".
    if isinstance(e, KeyError):
        return True
    code = getattr(e, "response", {}).get("Error", {}).get("Code")
    return code in ("NoSuchKey", "404", "NotFound")
