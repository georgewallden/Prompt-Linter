# modules/s3_bucket/main.tf

# This resource block creates the S3 bucket itself.
resource "aws_s3_bucket" "this" {
  bucket = var.bucket_name
  tags   = var.tags
}

# This resource block manages the versioning settings for the bucket created above.
resource "aws_s3_bucket_versioning" "this" {
  # This is a clever trick to make this resource conditional.
  # If var.enable_versioning is true, count is 1, and the resource is created.
  # If it's false, count is 0, and this block is completely ignored.
  count  = var.enable_versioning ? 1 : 0

  # This links the versioning setting to the bucket we just created.
  bucket = aws_s3_bucket.this.id

  versioning_configuration {
    status = "Enabled"
  }
}