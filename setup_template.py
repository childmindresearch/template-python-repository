#!/usr/bin python3
import dataclasses
import os
import pathlib as pl
from datetime import datetime
from typing import Optional

DIR_REPO = pl.Path(__file__).parent


@dataclasses.dataclass
class LicenseInfo:
    """A class representing information about a software license.

    Attributes:
        name: The name of the license.
        year : The year the license was issued.
        license_holder: The name of the entity holding the license.
    """

    path: pl.Path
    year: Optional[int] = None
    license_holder: str = ""

    @property
    def name(self) -> str:
        """Returns the name of the license."""
        return self.path.name.split("_", 1)[1]


def get_licenses() -> list[pl.Path]:
    """Returns a list of paths to license files in the 'licenses' directory.

    Returns:
         A list of paths to license files.

    """
    license_dir = pl.Path(__file__).parent / "licenses"
    return list(license_dir.glob("LICENSE_*"))


def request_license() -> Optional[LicenseInfo]:
    """Asks the user to select a license.

    Returns:
        The path to the selected license.

    """
    license_paths = get_licenses()
    license_options = [p.name.split("_", 1)[1] for p in license_paths]
    print("Available licenses:")
    print("\t0. Unlicensed")
    for i, option in enumerate(license_options):
        print(f"\t{i + 1}. {option}")
    while True:
        try:
            choice = int(input("Enter the number of the license you want to use: "))
            if choice == 0:
                return None
            if 0 < choice <= len(license_paths):
                repo_license = license_paths[choice - 1]
                break
            raise ValueError("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid choice. Please try again.")

    if repo_license.name == "LICENSE_MIT":
        license_holder = input("Who is the holder of the license? ")
        current_year = datetime.now().year
        return LicenseInfo(repo_license, current_year, license_holder)
    return LicenseInfo(repo_license)


def replace_license(repo_license: Optional[LicenseInfo]) -> None:
    """Replaces the license file in the repository with the specified license.

    Args:
        repo_license: The license to replace the current license with. If None, the current
            license file will be deleted.

    """
    license_file = DIR_REPO / "LICENSE"
    if not repo_license:
        license_file.unlink()
    else:
        repo_license.path.replace(license_file)

    for local_license in get_licenses():
        local_license.unlink()
    pl.Path(DIR_REPO / "licenses").rmdir()

    if not repo_license or repo_license.name != "MIT":
        return

    with open(license_file, "r", encoding="utf-8") as file_buffer:
        content = file_buffer.read()
    content = content.replace("[year]", str(repo_license.year))
    content = content.replace("[fullname]", repo_license.license_holder)

    with open(license_file, "w", encoding="utf-8") as file_buffer:
        file_buffer.write(content)


def main():
    # Collect some data
    git_uncommitted_changes = (
        os.popen(f"git -C {DIR_REPO} status -s").read().strip() != ""
    )
    git_username = os.popen(f"git -C {DIR_REPO} config user.name").read().strip()
    git_email = os.popen(f"git -C {DIR_REPO} config user.email").read().strip()
    git_repo_name = (
        os.popen(f"git -C {DIR_REPO} remote get-url origin")
        .read()
        .split("/")[-1]
        .split(".")[0]
    )

    # Ask for some data
    if git_uncommitted_changes:
        print("You have uncommitted changes. Please commit or stash them first.")
        exit(1)
    repo_name = (
        input(f"Enter the name of the repository [{git_repo_name}]: ") or git_repo_name
    )
    module_name = input(f"Enter the name of the module [{repo_name}]: ") or repo_name
    username = input(f"Enter your username [{git_username}]: ") or git_username
    email = input(f"Enter your email [{git_email}]: ") or git_email
    description = (
        input("Enter a short description of the project: ")
        or "A beautiful description."
    )
    repo_license = request_license()

    # Print the data
    print(
        f"Using the following values:\n"
        f"\tRepository name: '{repo_name}'\n"
        f"\tModule name: '{module_name}'\n"
        f"\tAuthor: '{username} <{email}'>\n"
        f"\tDescription: '{description}'\n"
        f"\tLicense: '{repo_license.name if repo_license else 'Unlicensed'}'"
    )
    input("Press enter to continue...")

    # Replace the template values
    for file in pl.Path(DIR_REPO).glob("**/*"):
        if (
            file.is_file()
            and not file.name == __file__
            and file.suffix in [".py", ".md", ".yml", ".yaml", ".toml", ".txt"]
        ):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()

            content_before = content
            content = content.replace(
                "- [ ] Run `setup_template.py`", "- [x] Run `setup_template.py`"
            )
            content = content.replace(
                "- [ ] Update the `LICENSE`", "- [x] Update the `LICENSE`"
            )
            content = content.replace("template-python-repository", repo_name)
            content = content.replace("APP_NAME", module_name)
            content = content.replace("app-name", module_name)
            content = content.replace("A beautiful description.", description)
            content = content.replace("reinder.vosdewael@childmind.org", email)
            content = content.replace("ENTER_YOUR_EMAIL_ADDRESS", email)
            content = content.replace("Reinder Vos de Wael", username)

            if not content == content_before:
                print(f"Updating {file.relative_to(DIR_REPO)}")
                with open(file, "w", encoding="utf-8") as f:
                    f.write(content)

    replace_license(repo_license)

    dir_module = DIR_REPO / "src" / "APP_NAME"
    if dir_module.exists():
        dir_module.rename(dir_module.parent / module_name)

    # Remove this file
    print(f"Removing {__file__}")
    pl.Path(__file__).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
