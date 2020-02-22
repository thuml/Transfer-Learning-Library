
def get_download_info(all_urls: dict, dependencies: list):
    """
    :param all_urls:
    :param dependencies:
    :return: A Triple  (file name, archive name, url link)
    """
    return [(filename, archive_name, all_urls[archive_name]) for filename, archive_name in dependencies]

