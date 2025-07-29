# terraform/backend.tf

terraform {
  backend "s3" {
    bucket = "prompt-linter-artifacts-us-east-1"
    key    = "terraform.tfstate" # The path/name for our state file in the bucket
    region = "us-east-1"
  }
}