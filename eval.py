from model import AdDetection

if __name__ == "__main__":
    ad_detect = AdDetection()
    ad_detect.fit_on_entire_training_set()
    ad_detect.evaluate()
