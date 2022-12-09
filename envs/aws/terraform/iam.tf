resource "aws_iam_instance_profile" "this" {
  name = "ml-system-design-22-iam"
  role = aws_iam_role.this.name
}

resource "aws_iam_role" "this" {
  name = "ml-system-design-22-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      },
    ]
  })

  tags = {
    Project = "ml-system-design-22"
  }
}

resource "aws_iam_role_policy" "this" {
  name = "ml-system-design-22-policy"
  role = aws_iam_role.this.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        "Effect" : "Allow",
        "Action" : [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ],
        "Resource" : [
          "arn:aws:s3:::${var.bucket_name}/*",
          "arn:aws:s3:::${var.bucket_name}"
        ]
      },
    ]
  })
}
