from helical.downloader import Downloader
if __name__ == "__main__":
    downloader = Downloader()
    downloader.get_ensemble_mapping('./data/21iT009_051_full_data.csv', './data/ensemble_to_display_name_batch_macaca.pkl')
    downloader.download_via_link("./data/33l_8ep_1024t_1280.torch", "https://figshare.com/ndownloader/files/43423236")