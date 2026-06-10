import boto3, json, os, sys, tarfile
import job_script
from error_reduction import error_mitigation

def setup_and_run(entry_point, input_file_key):
    """
    This method runs the user code.
    """
    print(f"Running container script with arguments: entry_point={entry_point}, input_file_key={input_file_key}")
    s3_client = boto3.client("s3")

    job_id = os.getenv("AWS_BATCH_JOB_ID")
    print(f"Job ID: {job_id}")

    s3_bucket_name = os.getenv("JOB_S3_BUCKET_NAME")
    output_file_name = os.getenv("JOB_OUTPUT_FILE_NAME")
    array_index = os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", default=None)
    print(f"Array index: {array_index}")

    # Download input params from S3
    params = {}
    if input_file_key:
        local_input = "input.json"
        print(f"Downloading input params from s3://{s3_bucket_name}/{input_file_key}")
        s3_client.download_file(s3_bucket_name, input_file_key, local_input)
        with open(local_input, "r") as f:
            params = json.load(f)

    # Run algorithm
    result = {}
    if entry_point == "test":
        result = job_script.run()
    elif entry_point == "error_mitigation":
        result = error_mitigation.run(params)

    # Save result to output
    output_path = os.path.join(os.getcwd(), output_file_name)
    print(f"Save result to output path: {output_path}")
    with open(output_path, "w") as f:
        json.dump(result, f)

    # Upload results to S3
    if array_index:
        parent_job_id, _, child_job_index = job_id.partition(":")
        print(f"Parent job id: {parent_job_id}, child job index: {child_job_index}, job array: {array_index}")
        s3_key = os.path.join("batch", parent_job_id, child_job_index, output_file_name)
    else:
        s3_key = os.path.join("batch", job_id, output_file_name)
    print(f"Upload result to S3 with object key {s3_key}")
    s3_client.upload_file(output_path, s3_bucket_name, s3_key)
    print("Job completed.")


if __name__ == "__main__":
    print(f"Command line arguments: {sys.argv}")
    setup_and_run(
        entry_point=(sys.argv[1]), 
        input_file_key=sys.argv[2] if len(sys.argv) > 2 else None
    )
