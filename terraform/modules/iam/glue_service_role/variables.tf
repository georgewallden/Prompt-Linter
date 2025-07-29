# terraform/modules/iam/glue_service_role/variables.tf

variable "role_name" {
  description = "The name for the Glue service IAM role."
  type        = string
}

variable "s3_raw_bucket_arn" {
  description = "ARN of the S3 bucket for raw data."
  type        = string
}

variable "s3_processed_bucket_arn" {
  description = "ARN of the S3 bucket for processed data."
  type        = string
}

variable "s3_artifacts_bucket_arn" {
  description = "ARN of the S3 bucket for script artifacts."
  type        = string
}

variable "tags" {
  description = "A map of tags to assign to the resources."
  type        = map(string)
  default     = {}
}