# AWS Credentials (optional S3 datasets)

AWS is entirely optional. Halo defaults to the HuggingFace Hub and local paths for datasets, and to a Docker image you build locally or pull anonymously from ECR Public; you only need AWS if you keep datasets (or checkpoints) in your own S3 bucket, or pull the image from your own ECR registry. In all cases you bring your own account, bucket, and credentials — nothing here is provisioned for you.

## Credential resolution

The S3 utilities in `src/data/sources/s3_client.py` build a `boto3.Session` with no explicit keys unless `aws_access_key_id` / `aws_secret_access_key` are passed, so credentials resolve through the standard boto3 chain: environment variables (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) → `~/.aws/credentials` → `~/.aws/config` (an SSO session) → an instance/role profile. No region is hardcoded in Python.

Any standard AWS setup works. The two common ones:

- **Access keys** — set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` (env vars or `~/.aws/credentials`). Simplest for CI, remote pods, and automation.
- **AWS SSO** — `aws configure sso` writes an `~/.aws/config` SSO session; refresh it with `aws sso login`. On headless machines use the legacy (non-`[sso-session]`) config format: it forces the device-code flow (`aws sso login --no-browser` prints a URL + user code to authorize from any device), while the `[sso-session]` format redirects to a localhost URI no headless machine can open.

Point the default S3 bucket at your own with `HALO_S3_DEFAULT_BUCKET=<your-bucket>`, or give fully-qualified `s3://<your-bucket>/...` dataset URIs.

## Credentials inside the training container

Mount `~/.aws` and pass the repo-root `.env` (it supplies `AWS_DEFAULT_REGION` alongside `WANDB_API_KEY` / `HF_TOKEN` and **must** be passed via `--env-file` — the code does not auto-load it):

```bash
SCRATCH=/mnt                     # your large volume — confirm with findmnt / df -h
docker run --rm --gpus all --env-file .env \
  -v $(pwd):/workspace -v "$SCRATCH:$SCRATCH" -v ~/.aws:/root/.aws -w /workspace \
  halo:blackwell bash -lc "..."
```

AWS credentials are never baked into the image. S3 datasets pre-cached under `$HALO_DATA_ROOT/s3_datasets` (keyed by `md5("<bucket>/<key>")`) load without any live AWS call — see [S3 utilities](../data/s3-utilities.md).

## ECR (optional private image registry)

If you host the Docker image in your own ECR registry rather than building it locally, authenticate Docker to it. The token expires after 12 hours:

```bash
aws ecr get-login-password --region <region> | \
    docker login --username AWS --password-stdin \
    <your-account>.dkr.ecr.<region>.amazonaws.com
```

For a hands-off setup install `amazon-ecr-credential-helper` and add your registry to `~/.docker/config.json` — it auto-refreshes tokens from your AWS credentials, no manual `docker login`:

```json
{
    "credHelpers": {
        "<your-account>.dkr.ecr.<region>.amazonaws.com": "ecr-login"
    }
}
```

To use ECR alongside Docker Hub, keep `credHelpers` for ECR and give Docker Hub a credential store too — `"credsStore": "pass"` (or `secretservice`) on Linux, `osxkeychain` on macOS. An `auths` block is the fallback for a registry with no helper: Docker writes it as base64 `username:password`, which anyone who can read the file recovers. Verify with `docker-credential-ecr-login list`. The VSCode Docker extension reads the same file, so it authenticates automatically once this is in place (reload the window after configuring).

Building the image from source (`make build-blackwell` / `make build-hopper`) needs no registry and no token, and the prebuilt images pull anonymously from ECR Public ([Docker → Registry](docker.md#registry)).

## Remote machines

Remote pods (RunPod, SkyPilot, multi-node workers) have no access to a local SSO session. Give each node its own credentials — export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` (or copy `~/.aws/` with `scp -r ~/.aws/ worker-node:~/`) — then run the ECR login above if pulling from a private registry. SkyPilot forwards your local AWS credentials to its instances automatically; the shipped tasks take the image through `resources.image_id`, not a `docker login` / `docker pull` in `setup:` — [SkyPilot → Docker image](skypilot.md#docker-image).

## Troubleshooting

| Error | Fix |
|-------|-----|
| `SSOTokenLoadError: The SSO session has expired` | `aws sso login` |
| ECR `unauthorized: authentication required` | Re-run `docker login`, or set up the credential helper |
| `NoCredentialsError: Unable to locate credentials` | Check the chain: `aws sts get-caller-identity`, `aws configure list` |
| `docker-credential-ecr-login: executable file not found` | `sudo apt-get install -y amazon-ecr-credential-helper` |
| `permission denied ... Docker daemon socket` | `sudo usermod -aG docker $USER` then `newgrp docker` |
| `ClientError: (AccessDenied)` on S3/ECR | Confirm the identity and its permissions: `aws sts get-caller-identity` |
| Headless SSO redirects to `http://127.0.0.1:...` | You are on the `[sso-session]` config format; switch to the legacy format so `aws sso login --no-browser` uses the device-code flow |

Never commit keys; keep `~/.aws/credentials` out of version control, rotate keys, and prefer SSO with least-privilege permissions.

Image tagging and publishing live in [Docker → Registry](docker.md#registry).
