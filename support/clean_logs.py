import os
import shutil


def delete_old_folders():
    folders = os.listdir("training/logs")

    if ".DS_Store" in folders:
        folders.remove(".DS_Store")

    folders.sort()

    for folder in folders[:-1]:
        if os.path.isdir("training/logs/" + folder):
            shutil.rmtree("training/logs/" + folder)

    print("\n", folders[-1], sep="")


if __name__ == "__main__":
    delete_old_folders()
