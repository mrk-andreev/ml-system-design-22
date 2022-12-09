resource "aws_s3_bucket" "this" {
  bucket = var.bucket_name

  tags = {
    Project = "ml-system-design-22"
  }
}

resource "aws_s3_bucket_acl" "this" {
  bucket = aws_s3_bucket.this.id
  acl    = "private"
}
