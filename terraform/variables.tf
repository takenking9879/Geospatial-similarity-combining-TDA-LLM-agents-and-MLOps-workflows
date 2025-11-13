variable "aws_region" {
  description = "Región de AWS donde se desplegarán los recursos"
  type        = string
  default     = "us-east-2"
}

variable "ami_id" {
  description = "ID de la AMI para la instancia EC2"
  type        = string
  default = "ami-0cfde0ea8edd312d4"
}

variable "instance_type" {
  description = "Tipo de instancia EC2"
  type        = string
  default     = "m7i-flex.large"
}

variable "key_pair_name" {
  description = "Nombre del par de llaves para la instancia EC2"
  type        = string
  default     = "my-ec2-key"
}

variable "security_group_name" {
  description = "Nombre del Security Group"
  type        = string
  default     = "my-app-sg"
}

variable "app_port" {
  description = "Puerto de la aplicación"
  type        = number
  default     = 8000
}

variable "ecr_repo_name" {
  description = "Nombre del repositorio ECR"
  type        = string
  default     = "tdagenai"
}