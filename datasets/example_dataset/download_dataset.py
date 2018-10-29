import os
import tarfile

from google_drive_downloader import GoogleDriveDownloader as gdd

def download_dataset(dest_path, dataset):
    tar_path = os.path.join(dest_path, dataset) + '.tar'
    gdd.download_file_from_google_drive(file_id='1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C',
                                        dest_path=tar_path, overwrite=True,
                                        unzip=False)

    tar = tarfile.open(tar_path)
    tar.extractall(dest_path)

    return
