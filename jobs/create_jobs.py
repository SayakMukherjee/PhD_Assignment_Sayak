#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 21-Feb-2023
# 
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

import re
import click

from pathlib import Path

def create_job_file(config_file: str, export_path: Path):
    """Create the slurm job file

    Args:
        config_file (str): configuration file
    """

    template_path = Path('pytorch.sbatch') # Template file

    lines = []

    with open(template_path, 'r') as fp:
        for line in fp:
            lines.append(line)

    updated_lines = []

    for line in lines:
        if re.search("output", line):
            updated_lines.append(line.replace("slurm", config_file))

        elif re.search("exp_config", line):
            updated_lines.append(line.replace("config_file", config_file))

        else:
            updated_lines.append(line)

    with open(export_path, 'w') as fp:
        for line in updated_lines:
            fp.write(line)

    return

@click.command()
@click.option('--overwrite', type=bool, default=False,
              help='Overwrite exsisting job files (default: False).')
def main(overwrite):
    """Generate job files for available configurations

    Args:
        overwrite (bool): overwrites any existing job files
    """


    config_dir = Path('../scripts')

    file_names = [x.stem for x in config_dir.iterdir() if x.is_file()]

    if 'template' in file_names:
        file_names.remove('template')

    for config_file in file_names:
        export_path = Path(config_file + '.sbatch')
        if not Path.exists(export_path) or overwrite:
            create_job_file(config_file, export_path)

if __name__ == '__main__':
    main()