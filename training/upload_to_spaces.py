import argparse
import os
from pathlib import Path

import boto3
from botocore.config import Config


def get_s3_client():
    endpoint = os.getenv("DO_SPACES_ENDPOINT", "https://sgp1.digitaloceanspaces.com")
    key = os.getenv("DO_SPACES_KEY")
    secret = os.getenv("DO_SPACES_SECRET")
    bucket = os.getenv("DO_SPACES_BUCKET", "nyapsys-models")

    if not key or not secret:
        raise ValueError("DO_SPACES_KEY and DO_SPACES_SECRET must be set")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )

    return s3, bucket


def upload_file(file_path: Path, prefix: str = "models/"):
    s3, bucket = get_s3_client()
    key = f"{prefix}{file_path.name}"

    print(f"Uploading {file_path} to s3://{bucket}/{key}...")

    s3.upload_file(
        str(file_path),
        bucket,
        key,
        ExtraArgs={
            "ContentType": "application/octet-stream",
            "ACL": "private",
        },
    )

    print(f"Uploaded: https://{bucket}.{s3.meta.endpoint_url.split('://')[1]}/{key}")


def upload_directory(source_dir: Path, prefix: str = "dataset/"):
    s3, bucket = get_s3_client()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for file_path in source_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(source_dir)
            key = f"{prefix}{rel_path}"

            print(f"Uploading {file_path} to s3://{bucket}/{key}...")

            s3.upload_file(
                str(file_path),
                bucket,
                key,
                ExtraArgs={
                    "ContentType": "application/octet-stream",
                    "ACL": "private",
                },
            )

    print("Upload complete")


def main():
    parser = argparse.ArgumentParser(description="Upload to DO Spaces")
    parser.add_argument(
        "--file",
        type=Path,
        help="Single file to upload (e.g., GGUF model)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Directory to upload (e.g., prepared dataset)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="S3 prefix/key prefix",
    )

    args = parser.parse_args()

    if args.file:
        upload_file(args.file, args.prefix or "models/")
    elif args.source:
        upload_directory(args.source, args.prefix or "dataset/")
    else:
        parser.print_help()
        raise ValueError("Either --file or --source required")


if __name__ == "__main__":
    main()