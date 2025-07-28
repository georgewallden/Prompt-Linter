# modules/s3_bucket/outputs.tf

output "bucket_id" {
  description = "The name (ID) of the S3 bucket."
  value       = aws_s3_bucket.this.id
}

output "bucket_arn" {
  description = "The Amazon Resource Name (ARN) of the S3 bucket."
  value       = aws_s3_bucket.this.arn
}