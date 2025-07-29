# terraform/modules/iam/glue_service_role/outputs.tf

output "role_arn" {
  description = "The ARN of the created Glue service role."
  value       = aws_iam_role.glue_role.arn
}