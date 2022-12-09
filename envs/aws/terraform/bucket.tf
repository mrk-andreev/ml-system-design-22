resource "aws_s3_bucket" "this" {
  bucket = "ml-system-design-22"

  tags = {
    Project = "ml-system-design-22"
  }
}

resource "aws_s3_bucket_acl" "this" {
  bucket = aws_s3_bucket.this.id
  acl    = "private"
}
