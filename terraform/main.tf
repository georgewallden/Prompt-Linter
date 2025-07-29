# terraform/main.tf      

# Configure provider to be AWS
provider "aws" {
  region = var.aws_region
}

# Define common variables and tags for the project
locals {
  buckets = {
    "raw-data"       = "data-raw"
    "processed-data" = "data-processed"
    "artifacts"      = "artifacts"
  }
  common_tags = {
    Project = var.project_name
    Purpose = "DataPipeline" # A general purpose for these resources
  }
}

# -----------------------------------------------------------------------------
# S3 BUCKETS
# Call the custom s3_bucket module and loop through our buckets map.
# -----------------------------------------------------------------------------
module "project_s3_buckets" {
  for_each = local.buckets

  source = "./modules/s3_bucket"

  bucket_name       = "${var.project_name}-${each.value}-${var.aws_region}"
  enable_versioning = true
  tags = merge(local.common_tags, {
    # Override the general purpose with a more specific one for each bucket
    Purpose = each.key
  })
}

# -----------------------------------------------------------------------------
# GLUE ETL JOB FOR DATA PROCESSING
# -----------------------------------------------------------------------------

# First, create the specialized IAM Role by calling our custom module.
module "prompt_linter_glue_role" {
  source = "./modules/iam/glue_service_role"

  role_name                 = "PromptLinterGlueJobRole"
  # We access the specific bucket ARNs from the module map created by the for_each loop.
  s3_raw_bucket_arn         = module.project_s3_buckets["raw-data"].bucket_arn
  s3_processed_bucket_arn   = module.project_s3_buckets["processed-data"].bucket_arn
  s3_artifacts_bucket_arn   = module.project_s3_buckets["artifacts"].bucket_arn
  tags                      = local.common_tags
}

# Second, create the Glue Job itself by calling our other custom module.
module "prompt_linter_etl_job" {
  source = "./modules/glue/glue_python_shell_job"

  job_name        = "prompt-linter-etl"
  # Link the job to the role created above.
  role_arn        = module.prompt_linter_glue_role.role_arn
  # Construct the script path using the specific artifacts bucket ID.
  script_location = "s3://${module.project_s3_buckets["artifacts"].bucket_id}/glue-scripts/glue_job.py"
  tags            = local.common_tags
}