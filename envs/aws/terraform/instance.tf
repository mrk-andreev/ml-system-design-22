resource "aws_instance" "this" {
  ami                    = var.ami
  instance_type          = var.instance_type
  vpc_security_group_ids = [aws_security_group.this.id]

  root_block_device {
    volume_size = var.instance_volume_size
  }

  user_data = <<-EOF
                #!/bin/bash
                sudo apt update -qq \
                  && sudo apt upgrade -yqq \
                  && sudo apt-get install -y docker.io \
                  && sudo curl -SL https://github.com/docker/compose/releases/download/v2.13.0/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose \
                  && sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose \
                  && sudo chmod +x /usr/bin/docker-compose \
                  && sudo snap install kubectl --classic \
                  && sudo snap install helm --classic
                EOF

  tags = {
    Project = "ml-system-design-22"
  }
}
