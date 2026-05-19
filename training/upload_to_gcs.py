import argparse
from pathlib import Path
from google.cloud import storage


def upload_to_gcs(file_path: str, bucket_name: str, destination_name: str = None):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    file_path = Path(file_path)
    destination = destination_name or file_path.name
    blob = bucket.blob(destination)
    print(f"Uploading {file_path} to gs://{bucket_name}/{destination}...")
    blob.upload_from_filename(str(file_path))
    print(f"Done: https://storage.googleapis.com/{bucket_name}/{destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--bucket", type=str, default="nyapsys-models")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    upload_to_gcs(args.file, args.bucket, args.name)


if __name__ == "__main__":
    main()