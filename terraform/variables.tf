# terraform/variables.tf

variable "project_name" {
  description = "The globally unique name for the project. Used as a prefix for resources."
  type        = string
  default     = "prompt-linter"
}

variable "aws_region" {
  description = "The AWS region to deploy all project resources into."
  type        = string
  default     = "us-east-1"
}