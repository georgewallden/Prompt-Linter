# terraform/modules/glue/glue_python_shell_job/variables.tf

variable "job_name" {
  description = "The name for the AWS Glue job."
  type        = string
}

variable "role_arn" {
  description = "ARN of the IAM role for the job to use."
  type        = string
}

variable "script_location" {
  description = "S3 path to the python script (e.g., s3://bucket/script.py)."
  type        = string
}

variable "python_version" {
  description = "The Python version for the job."
  type        = string
  default     = "3.9"
}

variable "libraries_to_install" {
  description = "A comma-separated list of libraries to install."
  type        = string
  default     = "pandas"
}

variable "tags" {
  description = "A map of tags to assign to the resources."
  type        = map(string)
  default     = {}
}