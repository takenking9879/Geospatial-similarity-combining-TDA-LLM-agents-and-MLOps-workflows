provider "aws" {
  region = var.aws_region
}

# --- Fuentes de datos ---
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# --- Recursos de AWS ---

## Repositorio ECR
resource "aws_ecr_repository" "tdagenai" {
  name = var.ecr_repo_name
}

## Security Group
resource "aws_security_group" "my_app_sg" {
  name        = var.security_group_name
  description = "Security group for application"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow HTTP"
  }
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow HTTPS"
  }
  ingress {
    from_port   = var.app_port
    to_port     = var.app_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow App Port"
  }
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow SSH"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = var.security_group_name
  }
}

## Generar el par de llaves localmente
resource "tls_private_key" "my_ec2_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "local_file" "my_ec2_key_pem" {
  filename        = "my-ec2-key.pem"
  content         = tls_private_key.my_ec2_key.private_key_pem
  file_permission = "0400"
}

resource "local_file" "my_ec2_key_pub" {
  filename = "my-ec2-key.pub"
  content  = tls_private_key.my_ec2_key.public_key_openssh
}

## Key Pair EC2
resource "aws_key_pair" "my_ec2_key" {
  key_name   = var.key_pair_name
  public_key = tls_private_key.my_ec2_key.public_key_openssh
}

## Instancia EC2 
resource "aws_instance" "my_app_instance" {
  ami                   = var.ami_id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.my_ec2_key.key_name
  vpc_security_group_ids = [aws_security_group.my_app_sg.id]
  subnet_id              = element(data.aws_subnets.default.ids, 0)

  root_block_device {
    volume_size = 30
  }

  # Script de inicialización al lanzar la instancia
  user_data = <<-EOF
              #!/bin/bash
              sudo apt-get update -y
              sudo apt-get upgrade -y
              curl -fsSL https://get.docker.com -o get-docker.sh
              sudo sh get-docker.sh
              sudo usermod -aG docker ubuntu
              newgrp docker
              EOF

  tags = {
    Name = "my-app-instance"
  }
}

# --- Outputs ---
output "ec2_public_ip" {
  value = aws_instance.my_app_instance.public_ip
}

output "ecr_uri" {
  value = aws_ecr_repository.tdagenai.repository_url
}

