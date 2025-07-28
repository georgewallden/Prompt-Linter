# terraform/main.tf  

# Configure provider to be AWS
provider "aws" {
  region = var.aws_region
}

# Map out our buckets
locals {
  buckets = {
    "raw-data"       = "data-raw"
    "processed-data" = "data-processed"
    "artifacts"      = "artifacts"
  }
}

# Call the custom s3_bucket module and run through all of our buckets
module "project_s3_buckets" {
  for_each = local.buckets

  # The 'source' tells Terraform where to find our reusable module's code.
  source = "./modules/s3_bucket"

  # ============================================================================
  # Here we pass the required INPUTS to our module for each of the 3 buckets.
  # ============================================================================

  # We construct the globally unique bucket name using our project variables.
  bucket_name = "${var.project_name}-${each.value}-${var.aws_region}"

  # We explicitly enable versioning for all our project buckets.
  enable_versioning = true

  # We assign project-specific tags to each bucket for organization.
  tags = {
    Project = var.project_name
    Purpose = each.key # 'each.key' refers to the key from our map, e.g., "raw-data".
  }
}