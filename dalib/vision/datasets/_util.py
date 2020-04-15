import os
from torchvision.datasets.utils import download_and_extract_archive


def download(root, file_name, archive_name, url_link):
    """ Download file from internet url link.
    :param root: (string) The directory to put downloaded files.
    :param file_name: (string) The name of the unzipped file.
    :param archive_name: (string) The name of archive(zipped file) downloaded.
    :param url_link: (string) The url link to download data.
    :return: None

    .. note::
    If `file_name` already exists under path `root`, then it is not downloaded again.
    Else `archive_name` will be downloaded from `url_link` and extracted to `file_name`.
    """
    if not os.path.exists(os.path.join(root, file_name)):
        print("Downloading {}".format(file_name))
        if os.path.exists(os.path.join(root, archive_name)):
            os.remove(os.path.join(root, archive_name))
        try:
            download_and_extract_archive(url_link, download_root=root, filename=archive_name, remove_finished=True)
        except Exception:
            print("Fail to download {} from url link {}".format(archive_name, url_link))
            print('Please check you internet connection or '
                  "reinstall DALIB by 'pip install --upgrade dalib'")
            exit(0)


def check_exits(root, file_name):
    """Check whether `file_name` exists under directory `root`. """
    if not os.path.exists(os.path.join(root, file_name)):
        print("Dataset directory {} not found under {}".format(file_name, root))
        exit(-1)
