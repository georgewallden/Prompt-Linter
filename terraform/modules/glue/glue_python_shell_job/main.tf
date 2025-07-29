# terraform/modules/glue/glue_python_shell_job/main.tf

resource "aws_glue_job" "python_shell_job" {
  name     = var.job_name
  role_arn = var.role_arn

  command {
    name            = "pythonshell"
    script_location = var.script_location
    python_version  = var.python_version
  }

  default_arguments = {
    "--library-set" = var.libraries_to_install
  }

  glue_version      = "3.0"
  tags              = var.tags
}