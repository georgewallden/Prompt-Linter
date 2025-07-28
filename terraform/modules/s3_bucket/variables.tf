# modules/s3_bucket/variables.tf

variable "bucket_name" {
  description = "The globally unique name of the S3 bucket."
  type        = string
}

variable "enable_versioning" {
  description = "A boolean flag to enable/disable versioning."
  type        = bool
  default     = false
}

variable "tags" {
  description = "A map of tags to assign to the bucket."
  type        = map(string)
  default     = {}
}