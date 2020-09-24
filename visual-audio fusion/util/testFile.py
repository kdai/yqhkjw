import numpy as np
from tqdm import tqdm
if __name__=='__main__':
    features = np.load('/home/share/Highlight/proDataset/TrainingSet/parkour_audio_edited.npy').tolist()
    keys = list(features.keys())
    delelte = []
    for key in tqdm(keys):

        feats = features[key]
        for feat in feats:
            f = feat['features']
            if not isinstance(f,list):
                print(key)
                print(f)
                delelte.append(key)
                break
    for d in delelte:
        del features[d]
    np.save('/home/share/Highlight/proDataset/TrainingSet/parkour_audio_edited.npy',features)

    # features = np.load('/home/share/Highlight/proDataset/TrainingSet/parkour_audio_raw.npy').tolist()
    # keys = list(features.keys())
    # delelte = []
    # for key in tqdm(keys):

    #     feats = features[key]
    #     for feat in feats:
    #         f = feat['features']
    #         if not isinstance(f,list):
    #             print(key)
    #             print(f)
    #             delelte.append(key)
    #             break
    # for d in delelte:
    #     del features[d]
    # np.save('/home/share/Highlight/proDataset/TrainingSet/parkour_audio_raw.npy',features)
    