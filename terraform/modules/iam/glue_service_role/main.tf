# terraform/modules/iam/glue_service_role/main.tf

resource "aws_iam_role" "glue_role" {
  name               = var.role_name
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "glue.amazonaws.com" }
    }]
  })
  tags = var.tags
}

resource "aws_iam_policy" "glue_policy" {
  name   = "${var.role_name}-Policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      // This statement for S3 is correct and stays the same
      {
        Sid      = "AllowS3Access"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = [
            "${var.s3_raw_bucket_arn}/*",
            "${var.s3_processed_bucket_arn}/*",
            "${var.s3_artifacts_bucket_arn}/glue-scripts/*"
        ]
      },
      // THIS IS THE NEW, CRITICAL STATEMENT FOR LOGGING
      {
        Sid      = "AllowCloudWatchLogs"
        Effect   = "Allow"
        Action   = [
            "logs:CreateLogGroup",
            "logs:CreateLogStream",
            "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*" // Standard practice for Glue logging
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "glue_attachment" {
  role       = aws_iam_role.glue_role.name
  policy_arn = aws_iam_policy.glue_policy.arn
}